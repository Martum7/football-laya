#!/bin/bash
# Lanca os 4 fetchers independentes em paralelo (fontes diferentes = sem interferencia).
# O limite e so nao ser bloqueado: cada script tem o seu rate limiter proprio.
# Uso: ./run_all.sh   |   ./run_all.sh stop
cd "$(dirname "$0")"
mkdir -p ../data/logs

if [ "$1" = "stop" ]; then
  pkill -f "fetch_sofascore.py" 2>/dev/null
  pkill -f "fetch_worldfootballr_data.py" 2>/dev/null
  pkill -f "fetch_understat.py" 2>/dev/null
  pkill -f "fetch_footballdata_odds.py" 2>/dev/null
  pkill -f "fetch_sorare.py" 2>/dev/null
  echo "stop enviado."
  exit 0
fi

# curl_cffi recomendado para o sofascore (senao o probe vai a 403 e o log diz isso)
python3 -c "import curl_cffi" 2>/dev/null || pip3 install -r requirements.txt

export PYTHONUNBUFFERED=1
# SORARE_API_KEY e GITHUB_TOKEN (opcional) vao por ambiente, nunca em ficheiro.

nohup python3 fetch_footballdata_odds.py > ../data/logs/odds.log 2>&1 &
echo "odds pid $!"
nohup python3 fetch_worldfootballr_data.py > ../data/logs/wfr.log 2>&1 &
echo "wfr pid $!"
nohup python3 fetch_understat.py > ../data/logs/understat.log 2>&1 &
echo "understat pid $!"
nohup python3 fetch_sofascore.py > ../data/logs/sofascore.log 2>&1 &
echo "sofascore pid $!"
# sorare fica em modo espera ate existir data/processed/players_wanted.jsonl
nohup python3 fetch_sorare.py --wait > ../data/logs/sorare.log 2>&1 &
echo "sorare(wait) pid $!"

echo "A acompanhar: tail -f ../data/logs/*.log"
