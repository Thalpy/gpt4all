import os
import openai
import discord
from discord.ext import commands

# GPT MODEL SETUP

# GPT-2

from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# GPT-J-6B

from transformers import pipeline

generator = pipeline('text-generation', model='EleutherAI/gpt-j-6B', device=0)

# OpenAI GPT-4

# get the openai api key from the openai api key file
with open("openai_token.txt", "r") as f:
    openai.api_key = f.read()

# END GPT MODEL SETUP

# Set up Discord bot
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_kongou_response_gptj(prompt):
    response = generator(prompt, max_length=1000, do_sample=True, temperature=0.7)

    return response

def generate_kongou_response_gpt2(prompt):
    full_prompt = f"Respond to the following message as if you were Kongou from Kancolle in a seductive mood: {prompt}"
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # remove the prompt from the response
    response = response.replace(full_prompt, "")
    print(response)
    return response.strip()

def generate_kongou_response_openai_gpt4(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Respond to the following message as if you were Kongou from Kancolle in a seductive mood: {prompt}",
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # Check if the bot is mentioned in the message
    if bot.user in message.mentions:
        prompt = message.content.replace(f"<@!{bot.user.id}>", "").strip()
        kongou_response = generate_kongou_response_openai_gpt4(prompt)
        await message.channel.send(kongou_response)

    await bot.process_commands(message)

# get the discord token from the discord token file
with open("discord_token.txt", "r") as f:
    discord_token = f.read()

bot.run(discord_token)
