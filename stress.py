import os
import pandas as pd
import numpy as np
from datetime import datetime
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)
from scipy.stats import linregress
import io
import matplotlib.pyplot as plt
from telegram import InputFile


CSV_FILE = "mood_log.csv"
QUESTIONS = [
    "On a scale from 1 to 5, how stressed did you feel today?",
    "On a scale from 1 to 5, how much energy did you have today?",
    "On a scale from 1 to 5, how productive were you today?"
]

# Initialize CSV if not present
def init_csv():
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=["timestamp", "stress", "energy", "productivity"])
        df.to_csv(CSV_FILE, index=False)

# Save today's answers
def append_entry(answers):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = pd.DataFrame([[timestamp] + answers], columns=["timestamp", "stress", "energy", "productivity"])
    entry.to_csv(CSV_FILE, mode='a', header=False, index=False)

# Start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["step"] = 0
    context.user_data["answers"] = []
    await update.message.reply_text("Hi! Let's track your day.\n" + QUESTIONS[0])

# Handle user responses to daily questions
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    step = context.user_data.get("step", 0)
    answers = context.user_data.get("answers", [])

    # Validate input
    text = update.message.text.strip()
    try:
        value = int(text)
        if not 1 <= value <= 5:
            raise ValueError
    except ValueError:
        await update.message.reply_text("Please enter a number from 1 to 5.")
        return

    answers.append(value)
    step += 1

    if step < len(QUESTIONS):
        context.user_data["step"] = step
        context.user_data["answers"] = answers
        await update.message.reply_text(QUESTIONS[step])
    else:
        append_entry(answers)
        await update.message.reply_text("Thanks! Your responses were saved.")
        context.user_data.clear()

# /stats command: daily mean values
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(CSV_FILE):
        await update.message.reply_text("No data yet.")
        return

    df = pd.read_csv(CSV_FILE)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    
    means = df.groupby("date")[["stress", "energy", "productivity"]].mean().round(2)

    response = "📊 Daily Averages (Last 7 Days):\n"
    for day in means.tail(7).itertuples():
        response += f"{day.Index}: Stress: {day.stress}, Energy: {day.energy}, Productivity: {day.productivity}\n"

    await update.message.reply_text(response)

# /trend command: composite score trend analysis
async def trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(CSV_FILE):
        await update.message.reply_text("No data available.")
        return

    df = pd.read_csv(CSV_FILE)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    daily = df.groupby("date")[["stress", "energy", "productivity"]].mean()
    if len(daily) < 4:
        await update.message.reply_text("Not enough data to detect a trend (need at least 4 days).")
        return
    
    ## Stress, energy, productivity
    ## C_min==-0.2 when scores are stress==5, energy== 1, productivity==1, C_max when 1,5,5
    daily["composite"] = (0.4*daily["energy"] + 0.4*daily["productivity"] - 0.2*daily["stress"]) 
    C_min, C_max = -0.2, 3.8    

    # absolute normalization
    daily["composite_norm"] = ((daily["composite"] - C_min) / (C_max - C_min)).clip(0, 1)
    
    
    # Check for trends

    last_3 = daily["composite_norm"].tail(3)
    ma = daily["composite_norm"].rolling(window=3).mean().dropna()

    if all(last_3 > ma.tail(3)):
        trend_msg = "📈 Uptrend detected in the last 3 days."
    elif all(last_3 < ma.tail(3)):
        trend_msg = "📉 Downtrend detected in the last 3 days."
    else:
        trend_msg = "🔄 No clear trend in the last 3 days."
    print(trend_msg)
    await update.message.reply_text(trend_msg)


async def regression_trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(CSV_FILE):
        await update.message.reply_text("No data available.")
        return

    df = pd.read_csv(CSV_FILE)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily = df.groupby("date")[["stress", "energy", "productivity"]].mean()
    if len(daily) < 7:
        await update.message.reply_text("Not enough data to do a regression test  (need at least 7 days).")
        return
    
    daily["composite"] = (0.4*daily["energy"] + 0.4*daily["productivity"] - 0.2*daily["stress"]) 
    
    ## Stress, energy, productivity
    ## C_min==-0.2 when scores are 5,1,1, C_max==3.8 when 1,5,5
    C_min, C_max = -0.2, 3.8  

    # absolute normalization
    daily["composite_norm"] = (daily["composite"] - C_min) / (C_max - C_min)

    window = 7
    y = daily["composite_norm"].tail(window).values
    x = np.arange(window)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    if p_value < 0.05:
        if slope > 0:
            trend_msg = f"📈 Significant uptrend\n r_value={round(r_value,2)}, p_value={round(p_value,4)}, std_err={round(std_err,4)}" 
        else:
            trend_msg= f"📉 Significant downtrend\n r_value={round(r_value,2)}, p_value={round(p_value,4)}, std_err={round(std_err,4)}"
    print(trend_msg)
    await update.message.reply_text(trend_msg)

async def plot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 1. Read & prepare data
    df = pd.read_csv(CSV_FILE)
    # df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["date"] = pd.to_datetime(
    df["timestamp"],
    format="mixed",
    errors="coerce",
    dayfirst=True
    ).dt.date
    daily = df.groupby("date")[["stress","energy","productivity"]].mean()
    # compute raw composite, normalize as before
    daily["composite_raw"] = (
        0.4*daily["energy"] + 
        0.4*daily["productivity"] - 
        0.2*daily["stress"]
    )
    C_min, C_max = -0.2, 3.8
    daily["composite_norm"] = (
        (daily["composite_raw"] - C_min) / (C_max - C_min)
    ).clip(0,1)

    # 2. Plot into a bytes buffer
    buf = io.BytesIO()
    plt.figure()
    plt.plot(daily.index, daily["composite_norm"])
    plt.xlabel("Date")
    plt.ylabel("Normalized Composite Score")
    plt.title("Daily Well-Being (Normalized Composite)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    # 3. Send the plot as a photo
    await context.bot.send_photo(
        chat_id=update.effective_chat.id,
        photo=buf,
        caption="Here’s your daily well-being over time!"
    )
    buf.close()
    print("Plot sent successfully.")


async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 1) Send the CSV as a document
    with open(CSV_FILE, "rb") as f:
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=InputFile(f, filename="mood_log.csv"),
            caption="Here’s your full mood log CSV."
        )

    # 2) Send a preview of the last 10 entries
    df = pd.read_csv(CSV_FILE)
    # Rename for readability
    preview = df.tail(10)[["timestamp", "stress", "energy", "productivity"]]
    text = "📋 Last 10 entries:\n" + preview.to_string(index=False)
    await update.message.reply_text(text)
    print("info sent successfully.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Get the user’s first name from Telegram
    name = update.effective_user.first_name or "there"

    help_text = (
        f"Hi, {name}! 🤖\n\n"
        "I’m your Personal Well-Being Tracker Bot. Here’s what I can do:\n\n"
        "• /start  — Ask you 3 quick daily questions (stress, energy, productivity) and save your answers.\n"
        "• /stats  — Show your daily average for each metric over the last 7 days.\n"
        "• /trend  — Tell you if your composite well-being score is trending up, down, or unclear over the last 3 days.\n"
        "• /plot   — Send you a time-series chart of your normalized composite score (0–1 scale).\n"
        "• /regression   — Regression-based trend test over last 7 days.\n"
        "• /info   — Download your full mood log CSV and preview the last 10 entries.\n"
        "• /help   — Display this message again.\n\n"
        "Use these commands every day to track how your stress, energy, and productivity\n"
        "change over time—and stay on top of your well-being! 🌟"
    )

    await update.message.reply_text(help_text)

# Main bot setup
def main():
    init_csv()

    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        print("⚠️ TELEGRAM_BOT_TOKEN is not set.")
        return

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("trend", trend))
    app.add_handler(CommandHandler("regression", regression_trend))
    app.add_handler(CommandHandler("plot", plot))
    app.add_handler(CommandHandler("info", info))
    app.add_handler(CommandHandler("help", help_command))


    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Bot is running.")
    app.run_polling()

if __name__ == "__main__":
    main()
   

# import pandas as pd

# # Sample data with some NaNs
# dates = pd.date_range("2025-05-17", periods=6, freq="D")
# values = [3, 4, None, 5, 6, None]
# df = pd.DataFrame({"composite": values}, index=dates)

# # Compute rolling means
# df["rolling_default"] = df["composite"].rolling(window=3).mean()
# df["rolling_min2"] = df["composite"].rolling(window=3, min_periods=2).mean()

# # Drop NaNs after rolling_default
# df["rolling_default_dropped"] = df["rolling_default"].dropna()

# # Display the DataFrame
# print(df)