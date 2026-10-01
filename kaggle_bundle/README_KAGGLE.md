# Kaggle — curva football-laya (T4 grátis)

O treino local (Mac) **continua a correr e não é mexido**. Isto é a via paralela no Kaggle.

## Conteúdo do bundle

- `dataset/` → para subir como Kaggle Dataset (826MB):
  `processed/{train,calib,test}.jsonl` (splits novos: train 16.019 / calib 1000 / test 1000),
  `models/laya-multilingual-base/` (pesos + tokenizer + configs),
  `scripts/{train_curve,eval_curve,config}.py` (idênticos aos locais),
  `dataset-metadata.json` (**editar o `id` com o teu username**).
- `notebook/curve_kaggle.ipynb` → importar no Kaggle (Create Notebook → Import).

## Passos (10 min, uma vez)

1. Conta Kaggle + verificação por telemóvel (exigida para GPU).
2. `pip install kaggle`, mete o `kaggle.json` (Account → Create New API Token) em `~/.kaggle/`.
3. Edita `dataset/dataset-metadata.json` → `id: "O_TEU_USERNAME/football-laya-curve"`.
4. Upload: `cd dataset && kaggle datasets create -p .`
   (atualizar depois: `kaggle datasets version -p . -m "v2"`)
5. Kaggle → Create → Notebook → **Import** `notebook/curve_kaggle.ipynb`.
6. Notebook Settings: Accelerator **GPU T4 x2**, Internet **ON**. Add data → o dataset do passo 4.
7. Corre as células por ordem. A célula de treino leva horas — é normal.

## Retoma e quota

- Sessões caem ao fim de ~9-12h: volta a correr a célula de treino, ela **retoma sozinha**
  (`resume.pt` de 200 em 200 steps; eval salta checkpoints prontos).
- Quota: ~30h GPU/semana (recomeça sábado 00:00 UTC). Grelha ≈ 20h.
- No fim: saca `data/processed/curve_kaggle/metrics.csv` e compara com o local
  (`data/processed/curve_oldsplit/metrics.csv` do piloto).

## Grelha no Kaggle

`2000, 5000, 10000, 16019` (todos os jogos — os N mais recentes do train, mesmo
antes do calib; calib/test fixos: 1000+1000 casos mais recentes) com
`--batch 32 --accum 1 --grad-checkpoint` — batch efetivo 32, igual ao piloto
local, por isso os números são comparáveis. Sem `--grad-checkpoint` a T4 16GB
pode não levar o backward de batch 32.

## Handoff entre contas (quando a quota de uma acaba)

Aviso: usar várias contas para contornar a quota pode violar os Termos do
Kaggle e dar ban (incluindo a conta principal). Decide informado.

1. Conta base: torna o dataset `football-laya-curve` **público**
   (página do dataset → Settings → Make public) para as outras o verem.
2. Conta A (quota a zeros): saca da sessão `data/models/curve_kaggle/` e
   `data/processed/curve_kaggle/` (Save Version ou download).
3. Monta pasta `handoff/` com `curve_kaggle_models/` (= conteúdo do 1º) e
   `curve_kaggle_metrics/` (= conteúdo do 2º); sobe como dataset
   `football-laya-progress` (privado chega) na conta B.
4. Conta B: importa o notebook, Add input do dataset base + do progress,
   corre as células por ordem **incluindo a de handoff** (aplica resumes e
   modelos prontos), depois treino (retoma sozinho) e eval (salta os prontos).
5. Repete para C/D/E. 5 × 30h ≈ 150h ≥ ~120h da grelha.
