import sys
import logging
import telebot
import requests
import json
from collections import defaultdict
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
from telebot.apihelper import ApiTelegramException
import io
from PIL import Image
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up logging to output to both file and console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("bot_errors.log"),
                        logging.StreamHandler(sys.stdout)
                    ])

# Configuration using environment variables
TELEGRAM_BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
GROQ_API_KEY = os.environ['GROQ_API_KEY']
OPENROUTER_API_KEY = os.environ['OPENROUTER_API_KEY']
TOGETHER_API_KEY = os.environ['TOGETHER_API_KEY']
HUGGING_FACE_API_KEY = os.environ['HUGGING_FACE_API_KEY']

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"  # Hugging Face API URL

YOUR_SITE_URL = "hadesbottelegram.com"
YOUR_SITE_NAME = "BoTElegramHades"

# Initialize bot
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# User data
user_conversations = defaultdict(list)
user_models = defaultdict(lambda: "llama-3.1-8b-instant")
user_characters = defaultdict(lambda: "default")

# Load available models and characters
GROQ_MODELS = [
    "llama-3.1-405b-reasoning", "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant", "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview", "llama3-70b-8192",
    "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it"
]

OPENROUTER_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4",
    "anthropic/claude-v1",
    "openai/gpt-4o-mini-2024-07-18",
    "gryphe/mythomax-l2-13b",
    "openai/gpt-4o-2024-08-06",
    "google/gemini-pro-1.5-exp",
    "perplexity/llama-3.1-sonar-small-128k-chat",
    "meta-llama/llama-3.1-8b-instruct:free",
]

TOGETHER_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
]


# Load characters with error handling
def load_characters():
    logging.info("Loading characters from JSON file...")
    try:
        with open('characters.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error("Error: characters.json file not found.")
        return {
            "default": {
                "name": "Default Character",
                "description": "A generic AI assistant."
            }
        }
    except json.JSONDecodeError:
        logging.error("Error: characters.json file is not valid JSON.")
        return {
            "default": {
                "name": "Default Character",
                "description": "A generic AI assistant."
            }
        }
    except Exception as e:
        logging.error(
            f"Unexpected error while reading characters.json: {str(e)}")
        return {
            "default": {
                "name": "Default Character",
                "description": "A generic AI assistant."
            }
        }


AVAILABLE_CHARACTERS = load_characters()

MODEL_EMOJIS = {
    "llama": "🦙",
    "mixtral": "🌪️",
    "gemma": "💎",
    "openai": "🔮",
    "anthropic": "🤖",
    "meta-llama": "🤖"
}


def show_error(message):
    logging.error(message)
    print(f"Error: {message}", file=sys.stderr)


def format_model_info(model_name):
    prefix = model_name.split(
        "/")[0] if "/" in model_name else model_name.split("-")[0]
    emoji = MODEL_EMOJIS.get(prefix, "🤖")
    return f"{emoji} Model: {model_name}"


def create_menu():
    markup = InlineKeyboardMarkup()
    markup.row_width = 2
    buttons = [
        InlineKeyboardButton("🔄 Reset", callback_data="reset"),
        InlineKeyboardButton("🔀 Groq Model",
                             callback_data="change_groq_model"),
        InlineKeyboardButton("🔀 OpenRouter Model",
                             callback_data="change_openrouter_model"),
        InlineKeyboardButton("🔀 Together AI Model",
                             callback_data="change_together_model"),
        InlineKeyboardButton("🎭 Karakter", callback_data="change_character"),
        InlineKeyboardButton("🖼️ Generate Image",
                             callback_data="generate_image"),
        InlineKeyboardButton("📊 Konteks", callback_data="context_count"),
        InlineKeyboardButton("💡 Ide Topik", callback_data="suggestions"),
        InlineKeyboardButton("❓ Bantuan", callback_data="help")
    ]
    markup.add(*buttons)
    return markup


def create_groq_model_menu():
    markup = InlineKeyboardMarkup()
    markup.row_width = 1
    buttons = [
        InlineKeyboardButton(format_model_info(model),
                             callback_data=f"groq_model_{model}")
        for model in GROQ_MODELS
    ]
    markup.add(*buttons)
    return markup


def create_openrouter_model_menu():
    markup = InlineKeyboardMarkup()
    markup.row_width = 1
    buttons = [
        InlineKeyboardButton(format_model_info(model),
                             callback_data=f"openrouter_model_{model}")
        for model in OPENROUTER_MODELS
    ]
    markup.add(*buttons)
    return markup


def create_together_model_menu():
    markup = InlineKeyboardMarkup()
    markup.row_width = 1
    buttons = [
        InlineKeyboardButton(format_model_info(model),
                             callback_data=f"together_model_{model}")
        for model in TOGETHER_MODELS
    ]
    markup.add(*buttons)
    return markup


def create_character_menu():
    markup = InlineKeyboardMarkup()
    markup.row_width = 2
    buttons = [
        InlineKeyboardButton(char_info['name'],
                             callback_data=f"character_{char_key}")
        for char_key, char_info in AVAILABLE_CHARACTERS.items()
    ]
    markup.add(*buttons)
    return markup


def create_suggestion_menu(suggestions):
    markup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    for suggestion in suggestions:
        markup.add(KeyboardButton(suggestion))
    markup.add(KeyboardButton("🔄 Generate Saran Baru"))
    return markup


@bot.message_handler(commands=['start'])
def send_welcome(message):
    logging.info("Handling /start command")
    welcome_text = (
        "👋 Halo! Saya adalah chatbot yang menggunakan Groq, OpenRouter, Together.ai, dan Hugging Face API.\n\n"
        "Silakan kirim pesan untuk memulai percakapan, atau ketik /menu untuk melihat opsi yang tersedia."
    )
    bot.reply_to(message, welcome_text)


@bot.message_handler(commands=['menu'])
def show_menu(message):
    logging.info("Handling /menu command")
    bot.send_message(message.chat.id, "Menu:", reply_markup=create_menu())


@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    logging.info(f"Handling callback query: {call.data}")
    handlers = {
        "reset": reset_conversation,
        "change_groq_model": show_groq_model_options,
        "change_openrouter_model": show_openrouter_model_options,
        "change_together_model": show_together_model_options,
        "change_character": show_character_options,
        "context_count": show_context_count,
        "suggestions": show_suggestions,
        "help": send_help,
        "generate_image": request_image_prompt
    }

    if call.data in handlers:
        handlers[call.data](call.message)
    elif call.data.startswith("groq_model_"):
        change_model(call.message, call.data.split("_", 2)[2], "groq")
    elif call.data.startswith("openrouter_model_"):
        change_model(call.message, call.data.split("_", 2)[2], "openrouter")
    elif call.data.startswith("together_model_"):
        change_model(call.message, call.data.split("_", 2)[2], "together")
    elif call.data.startswith("character_"):
        change_character(call.message, call.data.split("_", 1)[1])

    bot.answer_callback_query(call.id)
    try:
        bot.edit_message_reply_markup(call.message.chat.id,
                                      call.message.message_id,
                                      reply_markup=None)
    except ApiTelegramException as e:
        if "message is not modified" not in str(e):
            show_error(f"Telegram API error: {str(e)}")


def reset_conversation(message):
    logging.info("Resetting conversation")
    user_id = message.chat.id
    user_conversations[user_id] = []
    bot.send_message(user_id, "🔄 Riwayat percakapan Anda telah direset.")
    show_menu(message)


def show_groq_model_options(message):
    logging.info("Showing Groq model options")
    bot.send_message(message.chat.id,
                     "🔀 Pilih Groq model AI:",
                     reply_markup=create_groq_model_menu())


def show_openrouter_model_options(message):
    logging.info("Showing OpenRouter model options")
    bot.send_message(message.chat.id,
                     "🔀 Pilih OpenRouter model AI:",
                     reply_markup=create_openrouter_model_menu())


def show_together_model_options(message):
    logging.info("Showing Together AI model options")
    bot.send_message(message.chat.id,
                     "🔀 Pilih Together AI model:",
                     reply_markup=create_together_model_menu())


def show_character_options(message):
    logging.info("Showing character options")
    bot.send_message(message.chat.id,
                     "🎭 Pilih karakter bot:",
                     reply_markup=create_character_menu())


def request_image_prompt(message):
    logging.info("Requesting image prompt")
    bot.send_message(message.chat.id,
                     "Silakan kirim deskripsi gambar yang ingin Anda buat:")
    bot.register_next_step_handler(message, generate_image)


def generate_image(message):
    user_prompt = message.text
    logging.info(f"Generating image with prompt: {user_prompt}")
    bot.send_message(message.chat.id, "⏳ Sedang membuat gambar...")

    try:
        image_bytes = query_hugging_face({"inputs": user_prompt})
        image = Image.open(io.BytesIO(image_bytes))
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        bot.send_photo(message.chat.id,
                       photo=img_io,
                       caption="🖼️ Gambar yang dihasilkan:")
    except Exception as e:
        show_error(f"Error generating image: {str(e)}")
        bot.send_message(message.chat.id,
                         "❌ Maaf, terjadi kesalahan saat membuat gambar.")


def query_hugging_face(payload):
    logging.info("Querying Hugging Face API for image generation")
    response = requests.post(
        HUGGING_FACE_API_URL,
        headers={"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"},
        json=payload)
    response.raise_for_status()
    return response.content


def change_model(message, new_model, provider):
    logging.info(f"Changing model to {new_model} for provider {provider}")
    user_id = message.chat.id
    user_models[user_id] = new_model
    model_type = {
        "groq": "Groq",
        "openrouter": "OpenRouter",
        "together": "Together AI"
    }.get(provider, "Unknown")
    bot.send_message(
        user_id,
        f"✅ Model AI {model_type} diubah ke:\n{format_model_info(new_model)}")
    show_menu(message)


def change_character(message, new_character):
    logging.info(f"Changing character to {new_character}")
    user_id = message.chat.id
    user_characters[user_id] = new_character
    character_info = AVAILABLE_CHARACTERS[new_character]
    bot.send_message(user_id,
                     f"✅ Karakter diubah ke:\n{character_info['name']}")
    show_menu(message)


def show_context_count(message):
    logging.info("Showing context count")
    user_id = message.chat.id
    context_count = len(user_conversations[user_id])
    bot.send_message(message.chat.id,
                     f"📊 Jumlah pesan dalam konteks: {context_count}")
    show_menu(message)


def show_suggestions(message):
    logging.info("Showing conversation suggestions")
    user_id = message.chat.id
    current_model = user_models[user_id]
    current_character = user_characters[user_id]
    character_info = AVAILABLE_CHARACTERS[current_character]

    system_message = f"""You are {character_info['name']}, {character_info['description']}. 
    A user wants to chat with you. Generate 6 diverse, creative, and engaging conversation starters or questions that the user can ask you.
    Each suggestion should:
    1. Reflect your unique character and personality as {character_info['name']}.
    2. Be relevant to your background, expertise, or the time period you're from (if applicable).
    3. Encourage interesting and in-depth conversations.
    4. Be concise, not exceeding 15 words.
    5. Start with an appropriate emoji that matches the topic or tone of the suggestion.
    6. Be presented on a new line.

    Mix up the types of suggestions:
    - Thought-provoking questions for you to answer
    - Intriguing scenarios or "what if" situations related to your expertise or background
    - Topics for discussion that align with your interests or knowledge
    - Requests for you to share a story or anecdote related to your background

    Remember, these are suggestions for what the user might ask YOU, so phrase them accordingly.
    Stay true to your character's personality, knowledge, and era throughout all suggestions. in bahasa indonesia"""

    try:
        if current_model in OPENROUTER_MODELS:
            response = send_openrouter_request(system_message, current_model)
        elif current_model in TOGETHER_MODELS:
            response = send_together_request(system_message, current_model)
        else:
            response = send_groq_request(system_message, current_model)

        suggestions = response.strip().split('\n')
        suggestions = [s.strip() for s in suggestions if s.strip()][:6]
        bot.send_message(
            message.chat.id,
            f"💡 Berikut adalah beberapa pertanyaan yang bisa Anda ajukan kepada {character_info['name']}. Pilih salah satu, ketik pesan Anda sendiri, atau generate saran baru:",
            reply_markup=create_suggestion_menu(suggestions))
    except requests.exceptions.RequestException as e:
        error_message = f"Error generating suggestions: {str(e)}"
        show_error(error_message)
        bot.send_message(message.chat.id,
                         "❌ Maaf, tidak dapat menghasilkan saran saat ini.")
    except KeyError as e:
        error_message = f"Missing key in response: {str(e)}"
        show_error(error_message)
        bot.send_message(message.chat.id,
                         "❌ Maaf, tidak dapat menghasilkan saran saat ini.")
    except json.JSONDecodeError as e:
        error_message = f"JSON decode error: {str(e)}"
        show_error(error_message)
        bot.send_message(message.chat.id,
                         "❌ Maaf, tidak dapat menghasilkan saran saat ini.")


def send_help(message):
    logging.info("Sending help message")
    help_text = (
        "❓ Bantuan:\n\n"
        "• 💬 Kirim pesan untuk memulai percakapan\n"
        "• 🎤 Kirim pesan suara untuk transkripsi\n"
        "• 🔄 'Reset' untuk memulai percakapan baru\n"
        "• 🔀 'Groq Model' untuk ganti model Groq AI\n"
        "• 🔀 'OpenRouter Model' untuk ganti model OpenRouter AI\n"
        "• 🔀 'Together AI Model' untuk ganti model Together AI\n"
        "• 🎭 'Karakter' untuk ganti karakter bot\n"
        "• 🖼️ 'Generate Image' untuk membuat gambar dari deskripsi teks\n"
        "• 📊 'Konteks' untuk cek jumlah pesan\n"
        "• 💡 'Ide Topik' untuk mendapatkan ide percakapan\n"
        "• 🔍 Ketik /menu untuk menu utama")
    bot.send_message(message.chat.id, help_text)
    show_menu(message)


def handle_message(message, user_message=None):
    user_id = message.from_user.id
    user_message = user_message or message.text

    logging.info(f"Handling message from user {user_id}: {user_message}")

    current_character = user_characters[user_id]
    character_info = AVAILABLE_CHARACTERS[current_character]

    system_message = f"You are {character_info['name']}, {character_info['description']}. Respond accordingly."

    user_conversations[user_id].append({
        "role": "system",
        "content": system_message
    })
    user_conversations[user_id].append({
        "role": "user",
        "content": user_message
    })

    conversation_history = user_conversations[user_id][-6:]
    current_model = user_models[user_id]

    try:
        if current_model in OPENROUTER_MODELS:
            bot_response = send_openrouter_request(conversation_history,
                                                   current_model)
        elif current_model in TOGETHER_MODELS:
            bot_response = send_together_request(conversation_history,
                                                 current_model)
        else:
            bot_response = send_groq_request(conversation_history,
                                             current_model)

        user_conversations[user_id].append({
            "role": "assistant",
            "content": bot_response
        })
    except requests.exceptions.RequestException as e:
        error_message = f"Error processing request: {str(e)}"
        show_error(error_message)
        bot_response = "❌ Maaf, terjadi kesalahan saat memproses permintaan Anda."

    model_info = f"\n\n{format_model_info(current_model)}"
    character_info_text = f"\n{character_info['name']}"

    bot.reply_to(message, bot_response + model_info + character_info_text)


@bot.message_handler(
    func=lambda message: message.text == "🔄 Generate Saran Baru")
def regenerate_suggestions(message):
    logging.info("Regenerating suggestions")
    show_suggestions(message)


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    logging.info(f"Echoing message: {message.text}")
    if message.text != "🔄 Generate Saran Baru":
        handle_message(message)
    bot.send_message(message.chat.id,
                     "Ketik /menu untuk melihat opsi yang tersedia.")


def send_openrouter_request(messages, model):
    logging.info(f"Sending OpenRouter request with model: {model}")
    if isinstance(messages, str):
        messages = [{"role": "system", "content": messages}]

    try:
        response = requests.post(OPENROUTER_API_URL,
                                 headers={
                                     "Authorization":
                                     f"Bearer {OPENROUTER_API_KEY}",
                                     "HTTP-Referer": YOUR_SITE_URL,
                                     "X-Title": YOUR_SITE_NAME,
                                     "Content-Type": "application/json"
                                 },
                                 json={
                                     "model": model,
                                     "messages": messages,
                                     "temperature": 0.7,
                                     "max_tokens": 1024,
                                     "top_p": 1,
                                     "stream": False
                                 })
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        show_error(f"RequestException: {e}")
        raise
    except KeyError:
        show_error("Unexpected response structure.")
        raise


def send_groq_request(messages, model):
    logging.info(f"Sending Groq request with model: {model}")
    try:
        response = requests.post(
            GROQ_API_URL,  # Corrected URL
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 1,
                "stream": False
            })
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        show_error(f"RequestException during Groq API call: {e}")
        raise
    except KeyError:
        show_error("Unexpected response structure.")
        raise


def send_together_request(messages, model):
    logging.info(f"Sending Together request with model: {model}")
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    try:
        response = requests.post(TOGETHER_API_URL,
                                 headers={
                                     "Authorization":
                                     f"Bearer {TOGETHER_API_KEY}",
                                     "Content-Type": "application/json"
                                 },
                                 json={
                                     "model": model,
                                     "messages": messages,
                                     "temperature": 0.7,
                                     "max_tokens": 1024,
                                     "top_p": 1,
                                     "stream": False
                                 })
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        show_error(f"RequestException: {e}")
        raise
    except KeyError:
        show_error("Unexpected response structure.")
        raise


def main():
    logging.info("Starting the application...")

    try:
        bot.polling(none_stop=True)
    except ApiTelegramException as e:
        show_error(f"Telegram API error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
