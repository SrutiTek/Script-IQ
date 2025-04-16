import pandas as pd
import requests 
from bs4 import BeautifulSoup
import csv

import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from fredapi import Fred

from transformers import pipeline
import re
from tqdm import tqdm
from collections import Counter
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import math
import statsmodels.api as sm
from transformers import AutoTokenizer

from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
import itertools
import os
from dotenv import load_dotenv



load_dotenv()

# APIs
omdb_key = os.getenv("OMDB_API_KEY")
fred_key = os.getenv("FRED_API_KEY")
# List of movies you want to analyze (you can change or expand this list)

movies_list = list(set(["Inception", "Alien", "Avatar", "Wicked", "Frozen", "Whiplash", "X-Men", "Wild Wild West", "Tropic Thunder", "Saving Private Ryan", "17 Again", "Juno", "A Quiet Place", "Get Out", "Annie Hall", "500 Days of Summer", "2012", "Barbie", "Constantine", "Alone in the Dark", "Babel", "Life of Pi", "Dear White People", "Fight Club", 
                       "The Social Network", 
                        "The Blind Side", "Gladiator", "Pearl Harbor", "Joker", "Forrest Gump"])) 
#movies_list = list(set(["Inception", "Alien", "Avatar"])) 
nlp = spacy.load("en_core_web_md")
award_winners = set(["Whiplash", "Juno", "Get Out", "Forrest Gump", "Gladiator"])


##DATA COLLECTION
# Function to get movie data from OMDB API
def get_movie_data(movie_title, api_key):
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    response = requests.get(url)
    movie_data = response.json()
    
    if movie_data["Response"] == "True":
        return movie_data
    else:
        #print(f"Error fetching data for {movie_title}: {movie_data.get('Error')}")
        return None   
# Function to get Budget URL 
def get_budget_url(movie_title):
    search_url = f'https://www.boxofficemojo.com/search/?q={movie_title.replace(" ", "+")}'
    try:
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the first movie link in the search results
        movie_result = soup.find('a', href=True, string=movie_title)
        
        if movie_result:
            #print(f"Raw href: {movie_result['href']}") 
            movie_code = movie_result['href'].split('/')[2]  
            #print(f"Extracted movie code: {movie_code}")
            # Construct the URL
            movie_url = f"https://www.boxofficemojo.com/title/{movie_code}/?ref_=bo_se_r_1"
            #print(f"Constructed URL: {movie_url}")
            return movie_url

        
    except Exception:
        pass
    
    return None
# Function to get the budget amount
def get_budget(movie_url):
    if not movie_url:
        return None
    
    try:
        response = requests.get(movie_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract budget section
        budget_section = soup.find('span', string='Budget')
        
        if budget_section:
            budget_value = budget_section.find_next('span').text.strip().replace('$', '').replace(',', '')
            
            # Clean and convert to float if valid
            budget = float(budget_value) if budget_value.isdigit() else None
            return float(budget_value)
        
    except Exception:
        print(f"Error fetching budget")
    
    return None
# Function to get the movie script from IMSDb (scraping)
def get_movie_script(movie_title):
    #script_url = f"http://www.imsdb.com/scripts/{movie_title.replace(' ', '')}.html"
    #script_url = f"http://www.imsdb.com/scripts/{movie_title.replace(' ', '-').replace('\'', '')}.html" #remove apostrophes
    script_url = f"http://www.imsdb.com/scripts/{movie_title.replace(' ', '-')}.html"

    try:
        response = requests.get(script_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the <pre> tag where the script is typically found
        script_text = soup.find('pre')
        
        if script_text:
            cleaned_script = script_text.get_text()
            cleaned_script = re.sub(r'\n+', '\n', cleaned_script)  # Multiple line breaks → single line
            cleaned_script = re.sub(r'\s{2,}', ' ', cleaned_script)  # Multiple spaces → single space

            # Ensure consistent casing for character names
            cleaned_script = re.sub(r'(^[a-zA-Z0-9\s]+)(\n)', lambda m: m.group(1).upper() + m.group(2), cleaned_script, flags=re.M)

            # Return cleaned script
            return cleaned_script[:200000]  # Limiting to 100000 characters
        else:
            return "Script not found"
    except Exception as e:
        #print(f"Error fetching script for {movie_title}: {e}")
        return "Error fetching script"  
# Function to extract Rotten Tomatoes score
def get_rotten_tomatoes_score(movie_data):
    rt_score = next((rating['Value'] for rating in movie_data.get('Ratings', []) if rating['Source'] == 'Rotten Tomatoes'), 'N/A')
    return float(rt_score.rstrip('%')) / 10 if rt_score != 'N/A' else None
# Function to calculate Star Power
def calculate_star_power(imdb_rating, rt_score, cast):
    if imdb_rating is None and rt_score is None:
        return 0  # No data available

    imdb_rating = float(imdb_rating) if imdb_rating else 0
    rt_score = rt_score if rt_score else 0  # RT is already in 0-10 scale

    movie_score = imdb_rating + rt_score  # Hybrid score

    # Assign weights: Lead actor = 1.0, others = 0.5
    actors = [actor.strip() for actor in cast.split(",")] if cast else []
    #total_movies = len(actors) if actors else 1  # Avoid division by zero

    star_power_scores = {}
    for idx, actor in enumerate(actors):
        weight = 1.0 if idx == 0 else 0.5  # Lead role gets full weight
        if actor not in star_power_scores:
            star_power_scores[actor] = {"total_score": 0, "movie_count": 0}
        star_power_scores[actor]["total_score"] += movie_score * weight
        star_power_scores[actor]["movie_count"] += 1

    # Compute final Star Power scores
    return {actor: data["total_score"] / data["movie_count"] for actor, data in star_power_scores.items()}
def calculate_total_star_power(imdb_rating, rt_score, cast):
    # Get individual star power scores
    individual_star_power = calculate_star_power(imdb_rating, rt_score, cast)
    
    # Sum up the individual star power scores for all cast members
    total_star_power = sum(individual_star_power.values())
    
    return total_star_power
def fetch_cpi_data_fred(year):
    """
    Fetches the CPI value for a given year and 2025.
    Returns the inflation multiplier.
    """
    try:
        fred = Fred(api_key=FRED_API_KEY)

        # Get CPI for the movie's release year
        cpi_year = fred.get_series('CPIAUCSL', f"{year}-01-01", f"{year}-12-31").mean()

        # Get CPI for 2025
        cpi_2025 = fred.get_series('CPIAUCSL', '2025-01-01', '2025-12-31').mean()

        if cpi_year and cpi_2025:
            multiplier = cpi_2025 / cpi_year
            return round(multiplier, 4)

    except Exception as e:
        print(f"Error fetching CPI data for {year}: {e}")
    
    return 1.0  # Default to no inflation adjustment if API fails
def extract_release_year(movie_data):

    """Extracts the release year from the 'Released' field"""
    released = movie_data.get('Released', 'N/A')

    if released != 'N/A' and len(released.split()) >= 3:
        # Extract last four digits (year)
        year = released.split()[-1]
        if year.isdigit():
            return int(year)
    
    return None
# Function to check if movie exists in the CSV file
def movie_exists_in_csv(movie_title, filename='movie_data.csv'):
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                if row[0].strip().lower() == movie_title.strip().lower():
                    return True
    except FileNotFoundError:
        # If the file doesn't exist, it means this is the first movie being added
        return False
    return False  

##Script Features##
# Load RoBERTa model
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Extract dialogue from script using regex
# Improved dialogue extraction
def extract_dialogue(script):
    pattern = re.compile(r'\b([A-Z][A-Z\s\-]+)(?:\s*\(.*?\))?\s+(.*?)(?=(?:\b[A-Z][A-Z\s\-]+\b\s|\Z))', re.S)
    matches = pattern.findall(script)

    dialogues = []
    for character, line in matches:
        cleaned = line.strip().replace('\n', ' ')
        if cleaned:
            dialogues.append((character.strip(), cleaned))

    print(f"✅ Extracted {len(dialogues)} dialogue lines.")
    return dialogues

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
def truncate_text(text, max_tokens=512):
    input_ids = tokenizer.encode(text, truncation=True, max_length=max_tokens, add_special_tokens=True)
    return tokenizer.decode(input_ids, skip_special_tokens=True)

# Run sentiment analysis on each line
def analyze_sentiment(dialogues, batch_size=16):
    total_lines = len(dialogues)
    if total_lines == 0:
        return 0, {}, []

    def get_avg_sentiment_batched(lines):
        texts = [line for _, line in lines if line.strip()]
        scores = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch = [truncate_text(line) for line in batch]                 
            results = sentiment_pipeline(batch)
            for result in results:
                label = result['label']
                score = result['score']
                value = score if label == 'POSITIVE' else -score
                scores.append(value)
        return sum(scores) / len(scores) if scores else 0

    # ✅ Split into thirds for sentiment arc
    third = total_lines // 3
    first_third = dialogues[:third]
    second_third = dialogues[third:2*third]
    last_third = dialogues[2*third:]

    # 🧪 Script-level sentiment arc
    first_score = get_avg_sentiment_batched(first_third)
    second_score = get_avg_sentiment_batched(second_third)
    third_score = get_avg_sentiment_batched(last_third)
    script_sentiment_shift = abs(first_score - second_score) + abs(second_score - third_score)

    # Character-level sentiment (batched)
    character_scores = {}
    sentiment_log = []

    characters = [char for char, line in dialogues if line.strip()]
    lines = [line for char, line in dialogues if line.strip()]

    for i in range(0, len(lines), batch_size):
        batch_lines = lines[i:i+batch_size]
        batch_chars = characters[i:i+batch_size]
        batch_lines = [truncate_text(line) for line in batch_lines]  # <--- also truncate here
        results = sentiment_pipeline(batch_lines)

        for char, line, result in zip(batch_chars, batch_lines, results):
            label = result['label']
            score = result['score'] if label == 'POSITIVE' else -result['score']

            sentiment_log.append({
                "character": char,
                "line": line,
                "sentiment": label,
                "score": score
            })

            if char not in character_scores:
                character_scores[char] = []
            character_scores[char].append(score)

    def calculate_arc_complexity(scores):
        if len(scores) < 2:
            return 0
        return sum(abs(scores[i+1] - scores[i]) for i in range(len(scores)-1)) / (len(scores) - 1)
    
    character_arc_complexity = {
        char: calculate_arc_complexity(scores) for char, scores in character_scores.items()
    }
    top_characters = sorted(character_scores, key=lambda c: len(character_scores[c]), reverse=True)[:3]
    character_arc_complexity = {char: character_arc_complexity[char] for char in top_characters}

    return script_sentiment_shift, character_arc_complexity, sentiment_log


# --- Lexical Diversity ---
def compute_lexical_diversity(dialogues):
    all_words = " ".join([line for _, line in dialogues]).lower().split()
    total_words = len(all_words)
    unique_words = len(set(all_words))
    lexical_diversity = unique_words / total_words if total_words > 0 else 0
    return round(lexical_diversity, 4)

# --- Thematic Topics with LDA ---
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def extract_topics(dialogues, n_topics=5, n_words=10):
    # Combine all lines into a single list of strings
    lines = [line for _, line in dialogues if line.strip()]
    
    # Convert to document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(lines)

    # Run LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    # Get vocabulary
    feature_names = vectorizer.get_feature_names_out()

    # Extract top keywords per topic
    topic_keywords = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-n_words - 1:-1]
        keywords = [feature_names[i] for i in top_indices]
        topic_keywords.append(keywords)
    theme_keywords = {
        "love": ["love", "like", "relationship", "kiss", "heart", "date"],
        "friendship": ["friend", "buddy", "together", "trust", "hang", "group"],
        "betrayal": ["lie", "secret", "cheat", "betray", "deceive"],
        "family": ["dad", "mom", "sister", "brother", "home", "family"],
        "freedom": ["run", "escape", "free", "break", "leave"],
        "identity": ["name", "truth", "real", "who", "self", "believe"],
        "ambition": ["win", "goal", "dream", "try", "future", "career"],
        "conflict": ["fight", "argue", "battle", "war", "enemy", "kill"],
        "fear": ["scared", "afraid", "fear", "hide", "monster"],
        "sacrifice": ["sacrifice", "give", "cost", "save", "risk"],
        "war": ["battle", "war", "fight", "enemy", "soldier"],
        "survival": ["survive", "die", "kill", "safe", "danger"],
        "justice": ["court", "law", "judge", "justice", "truth"]
    }
    # Expand theme keywords using spaCy word vectors
    expanded_themes = {}
    for theme, base_words in theme_keywords.items():
        expanded = set()
        for word in base_words:
            token = nlp.vocab[word]
            if token.has_vector:
                expanded.update([
                    w for w in nlp.vocab
                    if w.has_vector and w.is_lower and w.prob >= -15
                    and token.similarity(w) > 0.6
                ])
        expanded_themes[theme] = expanded | set(base_words)

    # Match each topic to the most semantically similar theme
    named_themes = []
    for keywords in topic_keywords:
        best_theme = "Misc"
        max_matches = 0
        for theme, words in expanded_themes.items():
            matches = sum(1 for word in keywords if word in words)
            if matches > max_matches:
                max_matches = matches
                best_theme = theme
        named_themes.append(best_theme)

    return named_themes, topic_keywords


# saves data to movie_data.csv
def save_to_csv(movie_data, budget, sentiment_shift, top_char_arcs, lexical_diversity, named_themes, topic_keywords, indie_quality_score, filename='movie_data.csv'):
     # Check if the movie is already in the CSV file
    if movie_exists_in_csv(movie_data['Title'], filename):
        #print(f"Movie '{movie_data['Title']}' already exists in the CSV. Skipping.")
        return
    
    # Check if the CSV file exists. If not, create and write headers.
    file_exists = False
    try:
        with open(filename, 'r'):
            file_exists = True
    except FileNotFoundError:
        pass
    imdb_rating = movie_data.get('imdbRating', 'N/A')
    rt_score = get_rotten_tomatoes_score(movie_data)
    cast = movie_data.get('Actors', 'N/A')

    #Star power 
    star_power = calculate_star_power(imdb_rating, rt_score, cast)
    total_star_power = sum(star_power.values()) if star_power else 0

    # Boxoffice adjusted for time 

    release_year = extract_release_year(movie_data)
    box_office = movie_data.get('BoxOffice', 'N/A')

# Calculate the inflation multiplier
    multiplier = fetch_cpi_data_fred(release_year) if release_year else 1.0

# Adjust Box Office revenue for inflation
    adjusted_box_office = 'N/A'
    if box_office != 'N/A' and box_office.startswith('$'):
        box_office_value = float(box_office.replace('$', '').replace(',', ''))
        adjusted_box_office = round(box_office_value * multiplier, 2)
    #Append new movie data to the CSV
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        # Write headers only if the file is new
        if not file_exists:
            writer.writerow(['Title', 'Release Date', 'BoxOffice', 'BoxOffice(adj)', 'IMDb Rating', 'Rotten Tomatoes', 'Cast', 'Director', 'Budget', 'Star Power', 'Total Star Power', 'Sentiment Shift', 'ArcComplexity1', 'ArcComplexity2', 'ArcComplexity3', 'Lexical Diversity', 'Named Themes', 'Dominant Topics', 'Indie Quality Score'])
        
        #script = script.replace('"', '""')  
        # Write movie data to CSV
        writer.writerow([
            movie_data['Title'],
            movie_data.get('Released', 'N/A'),
            movie_data.get('BoxOffice', 'N/A'),
            adjusted_box_office,
            movie_data.get('imdbRating', 'N/A'),
            ', '.join([rating['Value'] for rating in movie_data.get('Ratings', []) if rating['Source'] == 'Rotten Tomatoes']),
            movie_data.get('Actors', 'N/A'),
            movie_data.get('Director', 'N/A'),
            budget if budget else 'N/A',  # Include budget,
            star_power, 
            total_star_power
            , 
            sentiment_shift, 
            top_char_arcs[0],
            top_char_arcs[1],
            top_char_arcs[2], 
            lexical_diversity,
            "; ".join(named_themes),
            "; ".join([", ".join(topic) for topic in topic_keywords]), 
            indie_quality_score 
        ])

def main():
    # Fetch and save movie data and scripts
    for movie_title in movies_list:
        print(f"\nProcessing movie: {movie_title}")
        
        if movie_exists_in_csv(movie_title, filename='movie_data.csv'):
            print(f"⏩ Skipping {movie_title} (already saved)")
            continue

        # Get metadata
        movie_data = get_movie_data(movie_title, api_key)
        if not movie_data:
            print(f"Skipping {movie_title} due to missing OMDB data.")
            continue

        # Get the cleaned script
        cleaned_script = get_movie_script(movie_title)

        # Only proceed if we have a valid script
        if cleaned_script and cleaned_script not in ["Script not found", "Error fetching script"]:
            dialogues = extract_dialogue(cleaned_script)
            sentiment_shift, character_arc_complexity, sentiment_log = analyze_sentiment(dialogues)
        
            top_char_arcs = [
            f"{char}: {round(score, 4)}"
            for char, score in character_arc_complexity.items()
            ]
            while len(top_char_arcs) < 3:
                top_char_arcs.append("N/A")

            # --- Lexical Diversity ---
            lexical_diversity = compute_lexical_diversity(dialogues)
            print(f"Lexical Diversity Score for {movie_title}: {lexical_diversity}")

            # Continue with topic modeling, sentiment analysis, etc.
            ...
        else:
            print(f"Skipping {movie_title} due to missing or bad script.")
            continue 

    # --- Thematic Topics ---
        named_themes, topic_keywords = extract_topics(dialogues)
        print(f"Top Topics for {movie_title}:")
        named_themes = named_themes[:5]
        while len(named_themes) < 5:
            named_themes.append("N/A")

        script_sentiment_shift, character_sentiments, sentiment_log = analyze_sentiment(dialogues)    
        print(f"Sentiment shift score for {movie_title}: {script_sentiment_shift}")
        ##CHARACTER DEPTH##
        # Count most frequent characters
        char_freq = Counter([entry["character"] for entry in sentiment_log])
        top_characters = [char for char, _ in char_freq.most_common(3)]

        # Build sentiment time series
        char_time_series = {char: [] for char in top_characters}
        for entry in sentiment_log:
            if entry["character"] in top_characters:
                char_time_series[entry["character"]].append(entry["score"])

        # Calculate arc complexity (std dev)
        top_char_arcs = []
        for char in top_characters:
            scores = char_time_series[char]
            std_dev = np.std(scores) if len(scores) > 1 else 0
            top_char_arcs.append(f"{char}: {round(std_dev, 4)}")

        # Pad to ensure 3 entries
        while len(top_char_arcs) < 3:
            top_char_arcs.append("N/A")

        # Budget
        movie_url = get_budget_url(movie_title)
        budget = get_budget(movie_url) if movie_url else None

        budget_value = budget if budget and budget > 0 else 1
        indie_quality_score = (script_sentiment_shift + lexical_diversity) / math.log(budget_value + 1)
        indie_quality_score = round(indie_quality_score, 4)

        print(f"🎬 Indie Quality Score for {movie_title}: {indie_quality_score}")
        # Save everything
        save_to_csv(movie_data, budget, script_sentiment_shift, top_char_arcs, lexical_diversity, named_themes, topic_keywords, indie_quality_score)


if __name__ == "__main__":
    main()



# ##BASIC ANALYSIS & VISUALIZATION##

csv_file = 'movie_data.csv'
movie_data = pd.read_csv(csv_file)
# Step 2: Print all column names
print("📄 ACTUAL COLUMNS IN CSV:")
print(movie_data.columns.tolist())

def clean_financial_data(df):
    # Remove $ and , from BoxOffice and convert to float
    if 'BoxOffice' in df.columns:
        df['BoxOffice'] = df['BoxOffice'].replace('[$,]', '', regex=True).astype(float, errors='ignore')

    if 'BoxOffice(adj)' in df.columns:
        df['BoxOffice(adj)'] = pd.to_numeric(df['BoxOffice(adj)'], errors='coerce')

    # Convert Rotten Tomatoes to numeric (assume it's already in % or float)
    if 'Rotten Tomatoes' in df.columns:
        df['Rotten Tomatoes'] = df['Rotten Tomatoes'].astype(str).str.replace('%', '', regex=False)
        df['Rotten Tomatoes'] = pd.to_numeric(df['Rotten Tomatoes'], errors='coerce')

    # Convert Budget to float if available
    if 'Budget' in df.columns:
        df['Budget'] = pd.to_numeric(df['Budget'], errors='coerce')

    # Convert Star Power columns
    if 'Star Power' in df.columns:
        df['Star Power'] = pd.to_numeric(df['Star Power'], errors='coerce')
    if 'Total Star Power' in df.columns:
        df['Total Star Power'] = pd.to_numeric(df['Total Star Power'], errors='coerce')

    # Convert Sentiment Shift and Lexical Diversity
    if 'Sentiment Shift' in df.columns:
        df['Sentiment Shift'] = pd.to_numeric(df['Sentiment Shift'], errors='coerce')
    if 'Lexical Diversity' in df.columns:
        df['Lexical Diversity'] = pd.to_numeric(df['Lexical Diversity'], errors='coerce')

    return df



df = clean_financial_data(movie_data)
print(movie_data.dtypes)


##LINEAR REGRESSION MODEL##

df = pd.read_csv('movie_data.csv')
df = clean_financial_data(df)

def parse_arc(arc):
    try:
        # Match a float at the end of the string after a colon
        match = re.search(r":\s*([\d.]+)$", arc)
        return float(match.group(1)) if match else np.nan
    except:
        return np.nan
df["Arc1"] = df["ArcComplexity1"].apply(parse_arc)
df["Arc2"] = df["ArcComplexity2"].apply(parse_arc)
df["Arc3"] = df["ArcComplexity3"].apply(parse_arc)
df['Avg Arc Complexity'] = df[['Arc1', 'Arc2', 'Arc3']].mean(axis=1)

def adjust_sentiment(s):
    try:
        if pd.isna(s):
            return np.nan
        # Ideal range = 0.18 to 0.22; peak = 0.20
        return max(0, 1 - abs(s - 0.20) * 10)  # scales from 1 down
    except:
        return np.nan

df['Sentiment_Adjusted'] = df['Sentiment Shift'].apply(adjust_sentiment)


def adjust_lexical(l, center=0.24, width=0.02):
    if pd.isna(l): return np.nan
    return 1 / (1 + np.exp(-(l - center) / width))
# Step 1: Apply sigmoid-style adjustment
df['Lexical_Adjusted'] = df['Lexical Diversity'].apply(lambda l: adjust_lexical(l, center=0.24, width=0.02))

# Step 2: Bin evenly across adjusted score range
bins = [0, 0.333, 0.667, 1.0]
df['Lexical_Adjusted_Bin'] = pd.cut(df['Lexical_Adjusted'], bins=bins)

# Step 3: Average Rotten Tomatoes per bin
lexical_rt_avg = df.groupby('Lexical_Adjusted_Bin')['Rotten Tomatoes'].mean()
print("\n🍅 RT by Lexical_Adjusted Bins (Centered Scoring):")
for bin_label, avg_val in lexical_rt_avg.items():
    print(f"   - {bin_label}: {avg_val:.2f}%" if not pd.isna(avg_val) else f"   - {bin_label}: No data")

df['Lexical_Adjusted'] = df['Lexical Diversity'].apply(adjust_lexical)

# ✅ REMOVE EXTREME RT OUTLIERS (5th–95th percentile)
q_low = df['Rotten Tomatoes'].quantile(0.05)
q_high = df['Rotten Tomatoes'].quantile(0.95)
df = df[(df['Rotten Tomatoes'] >= q_low) & (df['Rotten Tomatoes'] <= q_high)]

filtered_df = df[(df['Lexical_Adjusted'] > 0.333) & (df['Lexical_Adjusted'] <= 0.667)][['Title', 'Rotten Tomatoes']]
print("\n🎯 Movies in Lexical_Adjusted Bin 0.333 to 0.667:")
print(filtered_df)
bin_avg = filtered_df['Rotten Tomatoes'].mean()
print(f"\n📊 Corrected Average RT for 0.333–0.667 bin: {bin_avg:.2f}%")


required_columns = ['BoxOffice(adj)', 'Rotten Tomatoes', 'Sentiment Shift', 'Lexical Diversity', 'Avg Arc Complexity', 'Total Star Power', 'Budget']
df = df.dropna(subset=required_columns)
df['Award_Winner'] = df['Title'].apply(lambda x: x.strip().lower() in [t.lower() for t in award_winners])


df['Log_Budget'] = np.log1p(df['Budget'])  # log(1 + x) avoids log(0)
df['Log_BoxOffice'] = np.log1p(df['BoxOffice(adj)'])
# Normalize Rotten Tomatoes and Box Office
df['RT_Norm'] = df['Rotten Tomatoes'] / 100
df['BoxOffice_Norm'] = (df['Log_BoxOffice'] - df['Log_BoxOffice'].min()) / (df['Log_BoxOffice'].max() - df['Log_BoxOffice'].min())

# ✅ Calculate Weighted IQ Score FIRST
w_sentiment = .3
w_lexical = .2
w_arc = .5
print(f"\n🎯 Manually Tuned Weights Applied: Sentiment={w_sentiment}, Lexical={w_lexical}, Arc={w_arc}")
df['Weighted_IQ_Scores'] = (
    w_sentiment * df['Sentiment_Adjusted'] +
    w_lexical * df['Lexical_Adjusted'] +
    w_arc * df['Avg Arc Complexity']
)

min_score = df['Weighted_IQ_Scores'].min()
max_score = df['Weighted_IQ_Scores'].max()
df['Indie_Score_Normalized'] = (df['Weighted_IQ_Scores'] - min_score) / (max_score - min_score)

print("\n🎥 'Weighted_IQ_Scores':")
print(df[['Title', 'Weighted_IQ_Scores']])

avg_award = df[df['Award_Winner']]['Weighted_IQ_Scores'].mean()
avg_non_award = df[~df['Award_Winner']]['Weighted_IQ_Scores'].mean()
gap = avg_award - avg_non_award

# --- Requested Metrics ---
avg_arc_award = df[df['Award_Winner']]['Avg Arc Complexity'].mean()
avg_arc_non_award = df[~df['Award_Winner']]['Avg Arc Complexity'].mean()

sentiment_bins = pd.cut(df['Sentiment_Adjusted'], bins=3)
sentiment_boxoffice_avg = df.groupby(sentiment_bins)['Log_BoxOffice'].mean()

lexical_bins = pd.cut(df['Lexical_Adjusted'], bins=3)
lexical_rottentomatoes_avg = df.groupby(lexical_bins)['Rotten Tomatoes'].mean()

# --- Output ---
print(f"\n🎭 Average Arc Complexity:")
print(f"   🏆 Award-Winning Scripts: {avg_arc_award:.4f}")
print(f"   🎬 Other Scripts        : {avg_arc_non_award:.4f}")

print(f"\n💰 Average Box Office by Sentiment_Adjusted:")
for bin_label, avg_val in sentiment_boxoffice_avg.items():
    print(f"   - {bin_label}: {avg_val:.4f}")

print(f"\n🍅 Average Rotten Tomatoes by Lexical_Adjusted:")
for bin_label, avg_val in lexical_rottentomatoes_avg.items():
    print(f"   - {bin_label}: {avg_val:.2f}%")


features = ['Sentiment_Adjusted', 'Lexical_Adjusted', 'Avg Arc Complexity', 'Total Star Power', 'Log_Budget']
X = df[features]
X = sm.add_constant(X)  # add intercept
y = df['Log_BoxOffice']

model = sm.OLS(y, X).fit()
print(model.summary())

# Correlation heatmap
corr = df[['Weighted_IQ_Scores', 'Total Star Power', 'Budget']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Predictor Correlation Matrix")
plt.show()


# --- T-Test: Are award winners' character arc scores statistically higher? ---

award_arc_scores = df[df['Award_Winner']]['Avg Arc Complexity']
non_award_arc_scores = df[~df['Award_Winner']]['Avg Arc Complexity']

t_stat, p_val = ttest_ind(award_arc_scores, non_award_arc_scores, equal_var=False)

print("📊 T-Test Results (Avg Arc Complexity):")
print(f"T-Statistic: {round(t_stat, 4)}")
print(f"P-Value: {round(p_val, 4)}")
if p_val < 0.05:
    print("✅ Difference in Arc Complexity is statistically significant.")
else:
    print("⚠️ Difference is NOT statistically significant.")

# Optional: boxplot for Arc Complexity
sns.boxplot(x='Award_Winner', y='Avg Arc Complexity', data=df)
plt.title("Avg Arc Complexity: Award Winners vs Others")
plt.xlabel("Award Winner")
plt.ylabel("Avg Arc Complexity")
plt.show()

#💰 Regression Model 1: Predict Box Office (Adjusted)
 # Step 1: Define predictors and target
X_bo = df[['Sentiment Shift', 'Lexical Diversity', 'Avg Arc Complexity', 'Total Star Power', 'Log_Budget']]
y_bo = df['Log_BoxOffice']
X_bo = sm.add_constant(X_bo)

# Step 2: Run regression
model_bo = sm.OLS(y_bo, X_bo).fit()
print("----- Model 1: Predict Box Office (Adjusted) -----")
print(model_bo.summary())

# Step 3: Predict box office and calculate Commercial Alpha
df['Predicted_Log_BoxOffice'] = model_bo.predict(X_bo)
df['MonetaryUndervaluation'] = df['Predicted_Log_BoxOffice'] - df['Log_BoxOffice']

# Step 3b: Normalize to 0–1 scale
alpha_min = df['MonetaryUndervaluation'].min()
alpha_max = df['MonetaryUndervaluation'].max()
if alpha_max != alpha_min:
    df['MonetaryUndervaluation'] = 2 * (df['MonetaryUndervaluation'] - alpha_min) / (alpha_max - alpha_min) - 1
else:
    df['MonetaryUndervaluation'] = 0
df['MonetaryUndervaluation'] = 2 * (df['MonetaryUndervaluation'] - alpha_min) / (alpha_max - alpha_min) - 1

# Step 1: Convert log box office to actual $
df['Predicted_BoxOffice_$'] = np.expm1(df['Predicted_Log_BoxOffice'])
df['Actual_BoxOffice_$'] = np.expm1(df['Log_BoxOffice'])

# Step 2: Adjust for inflation (2020 → 2025 = ~1.20x)
CPI_MULTIPLIER = 1.20
df['Predicted_BoxOffice_2025_$'] = df['Predicted_BoxOffice_$'] * CPI_MULTIPLIER
df['Actual_BoxOffice_2025_$'] = df['Actual_BoxOffice_$'] * CPI_MULTIPLIER


df['RT_Normalized'] = df['Rotten Tomatoes'] / 100
df['CreativeUndervaluation'] = df['Weighted_IQ_Scores'] - df['RT_Normalized']

# Step 3: Verdict based on both undervaluation scores
def determine_verdict(row):
    if row['MonetaryUndervaluation'] > 0.5 and row['CreativeUndervaluation'] > 0.3:
        return "🎯 Hidden Gem (High Script, Undervalued)"
    elif row['MonetaryUndervaluation'] > 0.5:
        return "💰 Financially Undervalued"
    elif row['CreativeUndervaluation'] > 0.3:
        return "🧠 Critically Underrated Script"
    elif row['MonetaryUndervaluation'] < -0.5 and row['CreativeUndervaluation'] < -0.3:
        return "🚫 Overhyped & Underwhelming"
    else:
        return "✅ Fairly Valued"

df['Verdict'] = df.apply(determine_verdict, axis=1)

# Step 4: Show Top 10 Undervalued Films (Sorted by Monetary Undervaluation)
columns_to_display = [
    'Title',
    'Actual_BoxOffice_2025_$',
    'Predicted_BoxOffice_2025_$',
    'MonetaryUndervaluation',
    'CreativeUndervaluation',
    'Weighted_IQ_Scores',
    'Verdict'
]

top_combined = df.sort_values('MonetaryUndervaluation', ascending=False).head(10)

# Format dollar values (optional, for readability)
top_combined['Actual_BoxOffice_2025_$'] = top_combined['Actual_BoxOffice_2025_$'].apply(lambda x: f"${x:,.0f}")
top_combined['Predicted_BoxOffice_2025_$'] = top_combined['Predicted_BoxOffice_2025_$'].apply(lambda x: f"${x:,.0f}")

# Print final result
print("\n🎬 Final Report: Undervalued Films (w/ Verdicts)")
print(top_combined[columns_to_display].to_string(index=False))


# First, normalize Rotten Tomatoes score (0–100 → 0–1)
df['RT_Normalized'] = df['Rotten Tomatoes'] / 100

# Then, calculate Creative Undervaluation
df['CreativeUndervaluation'] = df['Weighted_IQ_Scores'] - df['RT_Normalized']

# Normalize both to 0–1 (optional but recommended for blending)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df[['MonetaryUndervaluation_Norm', 'CreativeUndervaluation_Norm']] = scaler.fit_transform(
    df[['MonetaryUndervaluation', 'CreativeUndervaluation']]
)

# # Blend into one final score
# df['UndervaluationScore'] = (
#     0.7 * df['MonetaryUndervaluation_Norm'] +
#     0.3 * df['CreativeUndervaluation_Norm']
# )

# Output: Show both normalized undervaluation scores
print("\n📊 Normalized Undervaluation Scores:")
print(df[['Title', 'MonetaryUndervaluation_Norm', 'CreativeUndervaluation_Norm']].sort_values(
    by='MonetaryUndervaluation_Norm', ascending=False))

 
# 📊 Visualize Predicted vs Actual Box Office
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Log_BoxOffice', y='Predicted_Log_BoxOffice')

# Perfect prediction reference line
plt.plot([df['Log_BoxOffice'].min(), df['Log_BoxOffice'].max()],
         [df['Log_BoxOffice'].min(), df['Log_BoxOffice'].max()],
         color='red', linestyle='--', label='Perfect Prediction')

plt.xlabel('Actual Log Box Office')
plt.ylabel('Predicted Log Box Office')
plt.title('Actual vs Predicted Box Office')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Sort all movies by undervaluation
all_undervalued = df.sort_values('MonetaryUndervaluation', ascending=False)

plt.figure(figsize=(12, max(6, 0.4 * len(all_undervalued))))  # Dynamically adjust height
sns.barplot(data=all_undervalued, y='Title', x='MonetaryUndervaluation', palette='viridis')
plt.xlabel('Monetary Undervaluation (Rescaled)')
plt.ylabel('Film Title')
plt.title('💰 All Films by Monetary Undervaluation')
plt.tight_layout()
plt.show()


correlation_matrix = df[['Sentiment_Adjusted', 'Lexical_Adjusted', 'Rotten Tomatoes']].corr()
print(correlation_matrix)

# Sentiment vs RT
sns.lmplot(x='Sentiment_Adjusted', y='Rotten Tomatoes', data=df)
plt.title('Sentiment vs Rotten Tomatoes')
plt.show()

# Lexical vs RT
sns.lmplot(x='Lexical_Adjusted', y='Rotten Tomatoes', data=df)
plt.title('Lexical Diversity vs Rotten Tomatoes')
plt.show()

import statsmodels.api as sm

X = df[['Sentiment_Adjusted', 'Lexical_Adjusted']]
X = sm.add_constant(X)
y = df['Rotten Tomatoes']

rt_model = sm.OLS(y, X).fit()
print(rt_model.summary())