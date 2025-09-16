# main.py
# Indian Market Bot (SmartAPI) — Manual Start/Stop Version
# Multi-TF (5m/15m/30m), GPT pattern analysis, charts, Telegram alerts
# Market time checks काढले आहेत, user manually start/stop करेल

import os, json, time, asyncio, traceback
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile

import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from dotenv import load_dotenv

try:
    from SmartApi import SmartConnect
except:
    from smartapi import SmartConnect

from openai import OpenAI
import aiohttp

load_dotenv()

# ---------------- CONFIG ----------------
SYMBOL_NAMES = ["NIFTY", "BANKNIFTY", "RELIANCE"]  # Friendly names
TF_MAP = {"5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE", "30m": "THIRTY_MINUTE"}

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 120))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SMARTAPI_API_KEY = os.getenv("SMARTAPI_API_KEY")
SMARTAPI_CLIENT_CODE = os.getenv("SMARTAPI_CLIENT_CODE")
SMARTAPI_PASSWORD = os.getenv("SMARTAPI_PASSWORD")
SMARTAPI_TOTP = os.getenv("SMARTAPI_TOTP")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

INSTR_MASTER_JSON_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
INSTR_CACHE_PATH = "/tmp/angel_instruments.json"
INSTR_CACHE_TTL = 24*3600

smartobj = None

# ---------------- Telegram ----------------
async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Not configured:", text[:120]); return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    await session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"})

async def send_photo(session, caption, path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(path,"rb") as f:
        data = aiohttp.FormData()
        data.add_field("chat_id",str(TELEGRAM_CHAT_ID))
        data.add_field("caption",caption)
        data.add_field("photo",f,filename="chart.png",content_type="image/png")
        await session.post(url,data=data)
    os.remove(path)

# ---------------- Instrument master ----------------
def download_instrument_master(force=False):
    if not force and os.path.exists(INSTR_CACHE_PATH):
        if time.time()-os.path.getmtime(INSTR_CACHE_PATH)<INSTR_CACHE_TTL:
            return json.load(open(INSTR_CACHE_PATH))
    r=requests.get(INSTR_MASTER_JSON_URL,timeout=20); r.raise_for_status()
    data=r.json(); json.dump(data,open(INSTR_CACHE_PATH,"w")); return data

def build_symbol_token_map(instr_list):
    m={}
    for i in instr_list:
        token=i.get("token"); symbol=(i.get("symbol") or "").upper(); name=(i.get("name") or "").upper()
        exch=i.get("exch_seg") or ""
        entry={"token":str(token),"symbol":i.get("symbol"),"name":i.get("name"),"exch":exch}
        if symbol: m.setdefault(symbol,[]).append(entry)
        if name: m.setdefault(name,[]).append(entry)
    return m

def lookup_token(instr_map,name):
    key=name.upper()
    if key in instr_map: return instr_map[key][0]
    for k,v in instr_map.items():
        if key in k: return v[0]
    return None

# ---------------- SmartAPI ----------------
async def smartapi_login():
    global smartobj
    if smartobj: return True
    def _login():
        s=SmartConnect(api_key=SMARTAPI_API_KEY)
        if SMARTAPI_TOTP: s.generateSession(SMARTAPI_CLIENT_CODE,SMARTAPI_PASSWORD,SMARTAPI_TOTP)
        else: s.generateSession(SMARTAPI_CLIENT_CODE,SMARTAPI_PASSWORD)
        return s
    loop=asyncio.get_running_loop()
    smart=await loop.run_in_executor(None,_login)
    if smart: smartobj=smart; return True
    return False

async def smartapi_get_candles(token,exch,interval,from_dt,to_dt):
    if not await smartapi_login(): return None
    params={"exchange":exch,"symboltoken":str(token),
            "interval":interval,"fromdate":from_dt.strftime("%Y-%m-%d %H:%M"),
            "todate":to_dt.strftime("%Y-%m-%d %H:%M")}
    def _call(): return smartobj.getCandleData(params)
    loop=asyncio.get_running_loop()
    resp=await loop.run_in_executor(None,_call)
    if not resp or not resp.get("status"): return None
    candles=[]; times=[]
    for r in resp["data"]:
        t=int(r[0])//1000; o,h,l,c,v=map(float,r[1:6])
        candles.append([o,h,l,c,v]); times.append(t)
    return {"candles":candles,"times":times}

# ---------------- GPT analysis ----------------
async def analyze_with_gpt(symbol,candles,tf):
    if not OPENAI_CLIENT or not candles: return None
    last=candles[-50:]; data="\n".join([f"{i}: O={o},H={h},L={l},C={c}" for i,(o,h,l,c,v) in enumerate(last)])
    prompt=f"Analyze {tf} candles for {symbol}. Identify candlestick & chart patterns. Give Bias."
    loop=asyncio.get_running_loop()
    def call(): return OPENAI_CLIENT.chat.completions.create(model=OPENAI_MODEL,
        messages=[{"role":"user","content":prompt+"\nData:\n"+data}],max_tokens=200,temperature=0.2)
    resp=await loop.run_in_executor(None,call)
    return resp.choices[0].message.content.strip()

# ---------------- Price Action ----------------
def compute_levels(candles,lookback=50):
    arr=candles[-lookback:] if candles else []
    if not arr: return (None,None,None)
    highs=sorted([c[1] for c in arr],reverse=True); lows=sorted([c[2] for c in arr])
    k=min(3,len(arr)); res=sum(highs[:k])/k; sup=sum(lows[:k])/k; mid=(res+sup)/2
    return sup,res,mid

def detect_signal(candles,tf):
    if not candles: return None
    last,prev=candles[-1],candles[-2]; sup,res,mid=compute_levels(candles)
    entry=last[3]; bias="NEUTRAL"; reason=[]; conf=50
    if res and entry>res: bias="BUY"; reason.append("Breakout"); conf+=15
    if sup and entry<sup: bias="SELL"; reason.append("Breakdown"); conf+=15
    if last[3]>last[0] and prev[3]<prev[0]: bias="BUY"; reason.append("Bullish Engulfing"); conf+=10
    if last[3]<last[0] and prev[3]>prev[0]: bias="SELL"; reason.append("Bearish Engulfing"); conf+=10
    sl=None; targets=[]
    if bias=="BUY": sl=min([c[2] for c in candles[-6:]])*0.997; targets=[entry+(entry-sl)*r for r in (1,2,3)]
    if bias=="SELL": sl=max([c[1] for c in candles[-6:]])*1.003; targets=[entry-(sl-entry)*r for r in (1,2,3)]
    return {"tf":tf,"bias":bias,"entry":entry,"sl":sl,"targets":targets,"reason":"; ".join(reason),"conf":conf,"levels":{"sup":sup,"res":res,"mid":mid}}

# ---------------- Chart ----------------
def plot_chart(tf_results,sym,trades):
    fig,axs=plt.subplots(3,1,figsize=(10,12),dpi=100)
    for i,(tf,ax) in enumerate(zip(TF_MAP.keys(),axs)):
        data=tf_results.get(tf,{}); candles=data.get("candles"); times=data.get("times")
        if not candles: continue
        dates=[datetime.utcfromtimestamp(t) for t in times]
        o=[c[0] for c in candles]; h=[c[1] for c in candles]; l=[c[2] for c in candles]; c_=[c[3] for c in candles]
        x=date2num(dates); width=0.6*(x[1]-x[0]) if len(x)>1 else 0.4
        for xi,oi,hi,li,ci in zip(x,o,h,l,c_):
            col="white" if ci>=oi else "black"
            ax.vlines(xi,li,hi,color="black"); ax.add_patch(plt.Rectangle((xi-width/2,min(oi,ci)),width,abs(ci-oi),facecolor=col,edgecolor="black"))
        trade=next((t for t in trades if t["tf"]==tf),None)
        if trade: ax.axhline(trade["entry"],color="blue"); ax.axhline(trade["sl"],color="red",linestyle="--")
        ax.set_title(f"{sym} {tf}")
    tmp=NamedTemporaryFile(delete=False,suffix=".png"); fig.savefig(tmp.name); plt.close(fig); return tmp.name

# ---------------- LOOP ----------------
async def loop():
    instr_list=download_instrument_master(force=True); instr_map=build_symbol_token_map(instr_list)
    resolved={n:lookup_token(instr_map,n) for n in SYMBOL_NAMES}
    async with aiohttp.ClientSession() as session:
        await send_text(session,"📈 Bot Started — Manual Mode")
        while True:
            try:
                for sym,info in resolved.items():
                    token=info.get("token"); exch=info.get("exch") or "NSE"
                    tf_results={}; trades=[]
                    for tf,interval in TF_MAP.items():
                        to_dt=datetime.now(); from_dt=to_dt-timedelta(minutes=110*(5 if tf=="5m" else (15 if tf=="15m" else 30)))
                        data=await smartapi_get_candles(token,exch,interval,from_dt,to_dt)
                        if data: tf_results[tf]=data; tr=detect_signal(data["candles"],tf)
                        if tr and tr["bias"]!="NEUTRAL" and tr["conf"]>=60:
                            gpt=await analyze_with_gpt(sym,data["candles"],tf); 
                            if gpt: tr["ai_patterns"]=gpt
                            trades.append(tr)
                    if trades:
                        text=f"🚨 *{sym}* Signals\n"+"\n".join([f"[{t['tf']}] {t['bias']} Entry={t['entry']} SL={t['sl']} Targets={t['targets']} | {t['reason']} | Conf={t['conf']}%" for t in trades])
                        chart=plot_chart(tf_results,sym,trades)
                        await send_text(session,text); await send_photo(session,text,chart)
                await asyncio.sleep(POLL_INTERVAL)
            except Exception as e:
                print("Loop error:",e); traceback.print_exc(); await asyncio.sleep(30)

if __name__=="__main__":
    asyncio.run(loop())
