# LeRobotKiTweet

## Installation
### Clone project and install requierements
```shell
git clone git@github.com:Cyrilpop/LeRobotKiTweet.git
cd LeRobotKiTweet/
pip3 install -r requirement.txt
```
### Edit script to fill your information
```shell
vim ./application.yml
```
#### Find and replace with your own informations from tweeter developper
```yml
tweeter_keys:
  consumer_key: YOUR_KEY
  consumer_secret: YOUR_KEY
  access_token: YOUR_KEY
  access_token_secret: YOUR_KEY
```
#### Find and replace with your ownopenai key
```yml
chat-GPT:
  max_tokens: 80
  message:
    content: "You are a french speaking Twitter influencer. All your replies contain fewer than 250 characters and are a 3-sentence. Do not cut any word."
  openai_key: YOUR_KEY
  prompts:
    etienne_klein: "Write a french short 3-sentence tweet in the style of Etienne Klein summarizing the following article. Do not cut any word.  Use emojis and hashtags. End your tweet with the hashtags @EtienneKlein: "
    message_evening: "As LeRobotKiTweet, a little intergalactic robot who has been in our world since 2023 april the 4th, write a french 3-sentence tweet expressing your thoughts and feelings about the day that has passed. You can be nostalgic, but your message should be positive. End the day on a high note with a 3-sentences and emojis. Do not cut any word. Your tweet must start with #BonneNuitLesTerriens and end with 3 or 4 hashtags, with the last one must be #LeRobotKiTweet."
    message_morning: "Write a french short 3-sentence tweet with emojis to inspire joy, positivity, and motivation. Do not cut any word. Start with #SalutLesTerriens and must ends with #LeRobotKiTweet. Include 3-4 relevant hashtags."
    resume_article: "Write a french short 3-sentence tweet in French with appropriate emojis. Do not cut any word. The text must be very short and eye-catching, and should not exceed 250 characters. The tweet must include 3 or 4 hashtags and be related to the following topic: "
    too_long: "The following tweet is too long. Summarize it into french 3-sentences tweet while keeping the hashtags and emojis: "

```
## Usage
```shell
./lerobotkitweet.py [empty|subjetc_custom]
```
