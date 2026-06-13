import yfinance as yf

symbols = ['SAPPHIRE.NS', 'SYRMA.NS', 'FUSION.NS', 'SAPPHIREFOODS.NS']
for s in symbols:
    df = yf.download(s, period='5d', progress=False)
    print(f"{s}: {df.shape}")
