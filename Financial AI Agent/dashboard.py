import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def show_dashboard(symbol, df, prediction, explanation):

    if explanation is None or explanation.strip() == "":
        explanation = "No AI explanation generated."

    root = tk.Tk()
    root.title("Financial AI Dashboard")
    root.geometry("1000x780")

    title = ttk.Label(
        root,
        text="Stock Analysis Dashboard - " + symbol,
        font=("Arial", 18, "bold")
    )
    title.pack(pady=10)

    latest = df.iloc[-1]

    current_price = float(latest["Close"])

    expected_return = (prediction - current_price) / current_price * 100

    if prediction > current_price:
        signal = "BUY"
        color = "green"
    else:
        signal = "SELL"
        color = "red"

    # -------------------
    # Stats Section
    # -------------------

    stats_frame = ttk.Frame(root)
    stats_frame.pack(pady=5)

    ttk.Label(stats_frame, text="Close: " + str(round(current_price, 2))).grid(row=0, column=0, padx=15)
    ttk.Label(stats_frame, text="SMA20: " + str(round(latest["SMA_20"], 2))).grid(row=0, column=1, padx=15)
    ttk.Label(stats_frame, text="RSI: " + str(round(latest["RSI"], 2))).grid(row=0, column=2, padx=15)
    ttk.Label(stats_frame, text="Volatility: " + str(round(latest["Volatility"], 4))).grid(row=0, column=3, padx=15)

    ttk.Label(stats_frame, text="Predicted Price: " + str(round(prediction, 2))).grid(row=1, column=0, padx=15)
    ttk.Label(stats_frame, text="Expected Return: " + str(round(expected_return, 2)) + "%").grid(row=1, column=1, padx=15)

    signal_label = tk.Label(
        stats_frame,
        text="Signal: " + signal,
        fg=color,
        font=("Arial", 12, "bold")
    )
    signal_label.grid(row=1, column=2, padx=15)

    # -------------------
    # Graph Section
    # -------------------

    graph_frame = ttk.Frame(root)
    graph_frame.pack(pady=10)

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(df.index, df["Close"], label="Close Price")
    ax.plot(df.index, df["SMA_20"], label="SMA 20")

    ax.set_title(symbol + " Price Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    plt.close(fig)

    # -------------------
    # Explanation Section
    # -------------------

    explain_label = ttk.Label(
        root,
        text="AI Financial Explanation",
        font=("Arial", 14, "bold")
    )
    explain_label.pack(pady=5)

    text_frame = tk.Frame(root)
    text_frame.pack(fill="both", expand=True, padx=20, pady=10)

    scrollbar = tk.Scrollbar(text_frame)
    scrollbar.pack(side="right", fill="y")

    text_box = tk.Text(
        text_frame,
        wrap="word",
        yscrollcommand=scrollbar.set,
        font=("Arial", 11),
        height=12
    )

    text_box.pack(fill="both", expand=True)
    scrollbar.config(command=text_box.yview)

    text_box.insert("1.0", explanation)
    text_box.config(state="disabled")

    ttk.Button(root, text="Close", command=root.destroy).pack(pady=10)

    root.mainloop()