# football-laya — fetch Big5 (max anos)

Pasta do projeto: `DEV/Versioned/Experiment/football-laya/`
- `scripts/` fetchers (cada fonte com o seu rate limiter proprio, correm em paralelo)
- `data/raw/` espelhos locais (resume: ficheiro existente = skip)
- `data/logs/` logs dos jobs em background
- `data/processed/` sai do build (train/test, players_wanted.jsonl) — ainda por fazer

## Estado (2026-10-03)

| Fonte | Estado | Notas |
|---|---|---|
| football-data odds (50 CSVs 2015–2024) | DONE | `data/raw/odds_footballdata/fd_{E0,SP1,D1,I1,F1}_{1516..2425}.csv` |
| worldfootballR_data (35 .rds Big5) | DONE | resultados ENG/ESP/GER/ITA/FRA + fotmob + understat_shots + tm + season stats |
| Understat scrape direto | DEPRECATED | site tirou o JSON embebido; coberto pelos .rds |
| Sofascore API direta | BLOQUEADA (403 Cloudflare até com impersonation) | despriorizada |
| FotMob `/api/data/*` (dias + matchDetails) | A CORRER em background | cobre lineups, stats, h2h, tabela, events |
| Sorare | em espera (`--wait`) | precisa `SORARE_API_KEY` + `players_wanted.jsonl` do build |
| Python 3.12 + pyreadr (ler .rds) | a instalar via brew | preciso para converter .rds → parquet |

## Correr / parar

```bash
cd DEV/Versioned/Experiment/football-laya/scripts
export SORARE_API_KEY="..."   # a tua key (200 req/min; usamos 180/min)
export GITHUB_TOKEN="..."     # opcional, evita rate-limit da API GitHub
./run_all.sh                  # lanca tudo em background
./run_all.sh stop             # para tudo
tail -f ../data/logs/*.log    # acompanhar
```

Jobs lançados a 2026-10-03 ~18:10: odds DONE, wfr DONE (35/35),
sorare em `--wait` (PID ver com `ps aux | grep fetch_sorare`).

## Chaves (.env)

Ficheiro: `DEV/Versioned/Experiment/football-laya/.env` (criado, vazio).
Cola la dentro:

```
SORARE_API_KEY=a-tua-key
```

O `fetch_sorare.py` le esse ficheiro sozinho. Variavel de ambiente real
tem prioridade sobre o `.env`. Nunca commitar com valores.

## Contrato anti-leakage (15 regras, trancadas)

- `fb_*_season_stats` = TOTAIS finais → nunca feature direta.
- Odds fora do `state_text` (so coluna `odds_at_cutoff` para EV).
- Tudo o resto: so jogos com data < `cutoff (kickoff-50min)`.
