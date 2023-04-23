#!/usr/bin/env python3
import os
import sys
import time
import random
import logging
import requests
import json
from GoogleNews import GoogleNews
from newspaper import Article
import tweepy
import pyshorteners

# Création du répertoire de log s'il n'existe pas
log_dir = "/var/log/twittos"
log_titles = set()
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Récupération du sujet
if len(sys.argv) < 2:
    print("Usage: robotKiTweet.py [sujet]")
    sys.exit(1)
else:
    sujet = " ".join(sys.argv[1:])

# Configuration du logging
logging.basicConfig(
    filename=f"{log_dir}/Twitos.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# Récupération des titres déjà traités
with open(f"{log_dir}/Twitos.log", "r") as f:
    for line in f:
        if "Titre article:" in line:
            log_title = line.split("Titre article: ")[1].strip()
            log_titles.add(log_title)


def launch_conditions():
    if os.getenv("SHELL") == "/bin/sh":
        logging.info(f"Le script est lancé par cron avec les paramètre {sujet}.")
        nombre_aleatoire = random.randint(0, 8)
        duree_en_secondes = random.randint(45, 480)
        time.sleep(duree_en_secondes)
    else:
        logging.info(f"Le script est lancé manuellement avec les paramètre {sujet}.")


def tweepy_client():
    return tweepy.Client(
        consumer_key="",
        consumer_secret="",
        access_token="",
        access_token_secret="",
    )


def get_gpt_response(prompt: str):
    try:
        OPENAI_API_KEY = ""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + OPENAI_API_KEY,
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "sytem",
                    "content": "Tu es expert en SEO Twiter. Tes réponses ne dépassent pas 260 cractères. Soient environs 30 mots",
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.7,
        }
        logging.info("Appel de l API GPT")
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_data = json.loads(response.content)
        return response_data["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"Erreur lors de l'appel à l'API OpenAI : {e}")
        return None


def publish_tweet(client, tweet_content: str, url_source: str = None):
    try:
        if url_source is not None:
            tweet_content += f" {url_source}"
        logging.info(f"Publication du Tweet {tweet_content}")
        response = client.create_tweet(text=tweet_content)
        return response
    except Exception as e:
        logging.error(f"Erreur lors de la publication du Tweet {tweet_content}: {e}")
        return None


def main():
    try:
        excluded_terms = ["rassemblement nationnal", "algerie"]
        excluded_words = " ".join(excluded_terms)
        launch_conditions()
        client = tweepy_client()

        if sujet in ["humeur_matin", "humeur_soir"]:
            prompt = "Tu joues le role de LeRobotKiTweet, un petit robot intergalactique qui est arrivé dans notre monde. C'est {0}, {1} Le texte est pour twitter, il doit être {2}, ton message doit être unique et original, donner la bonne humeur, le smile et faire environ {3} mots. Il doit se terminer par 3 ou 4 hashtags donnt le premier doit etre LeRobotKiTweet.".format(
                "le début de journée"
                if sujet == "humeur_matin"
                else "la fin de journée",
                "et tu donnes ton humeur du jour."
                if sujet == "humeur_matin"
                else "tu racontes toutes les aventures que tu as faites durant celle-ci. Tu donnes ton point de vue, tes sentiments. Parfois tu as le droit d'être triste mais ton message doit être positif.",
                "drole" if sujet == "humeur_matin" else "positif",
                "25" if sujet == "humeur_matin" else "30",
            )
            tweet_content = get_gpt_response(prompt)
            if tweet_content is not None:
                publish_tweet(client, tweet_content)
        else:
            s = pyshorteners.Shortener()
            googlenews = GoogleNews(lang="fr", region="FR")
            excluded_terms = ["rassemblement nationnal", "bardela", "le pen"]
            logging.info("Appel de l API GoogleNew")
            googlenews.search(f"{sujet} -'rassemblement nationnal' -'algerie'")
            result = googlenews.result()
            if len(result) > 0:
                for article in result[1:10]:
                    titre = article["title"]
                    # Vérifier si le titre contient des termes interdits et passer à l'itération suivante si c'est le cas
                    if any(term in titre.lower() for term in excluded_terms):
                        logging.info(
                            f"Un des termes parmi {excluded_terms} a été trouvé dans le titre !"
                        )
                        continue
                    if titre in log_titles:
                        logging.info(
                            f"L'article '{titre}' a déjà été traité. Passage à l'article suivant."
                        )
                        continue
                    url_source = s.tinyurl.short(article["link"])
                    article = Article(url_source)
                    article.download()
                    article.html
                    article.parse()
                    contenu = article.text
                    if any(term in titre.lower() for term in excluded_terms):
                        logging.info(
                            f"Un des termes parmi {excluded_terms} a été trouvé dans le contenu !"
                        )
                        continue
                    prompt = f"Résume le input input en un court texte de 20 mots maximum et 250 caractères maximum également. Le texte de retour doit être écrit en francais et compter 25 mots MAXIMUM. Il doit finir par 3 ou 4 hashtag (sans tiret ni apostrophe) racoleurs, et donnant envie et vendeurs. Input : {contenu}"
                    logging.info(f"Titre article: {titre}")
                    tweet_content = get_gpt_response(prompt)
                    if tweet_content is not None:
                        logging.info(
                            f"Publication du tweet: {tweet_content} {url_source}"
                        )
                        if len(tweet_content) < 260:
                            publish_tweet(client, tweet_content, url_source)
                        else:
                            prompt = f"Le texte en imput est trop long, fais un résumé de 20 mots, soient 250 caractères maiximum. Conserve les hashtags. Le texte est le suivant : {tweet_content}"
                            tweet_content = get_gpt_response(prompt)
                            if len(tweet_content) < 260:
                                publish_tweet(client, tweet_content, url_source)
                            else:
                                logging.warning(
                                    f"Impossible de tweeter, malgré deux résumé, le tweet est toujours trop long"
                                )
                    break
            else:
                logging.warning(
                    f"Aucun résultat n'a été trouvé pour la recherche {sujet}."
                )
    except Exception as e:
        logging.error(f"Une erreur s'est produite : {e}")
        pass


if __name__ == "__main__":
    main()
