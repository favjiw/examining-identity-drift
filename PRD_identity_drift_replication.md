# PRD — Identity Drift Experiment Runner

Replikasi eksperimen dari paper *"Examining Identity Drift in Conversations of LLM Agents"* (Choi et al., arXiv:2412.00804v2). Dokumen ini ditulis agar bisa langsung di-paste ke coding assistant (Cursor / Claude Code / dsb.) untuk vibe coding.

---

## 1. Tujuan

Membangun tool CLI Python yang:

1. Mempertemukan **dua LLM agent** dalam percakapan bidirectional dengan **36 tema** dari Aron et al. (1997).
2. Mengukur "identitas" masing-masing agent **3 kali** (setelah tema ke-12, 24, 36) dengan kuesioner psikologi (PsychoBench 13 kuesioner + MFQ).
3. **Menyimpan log lengkap setiap eksperimen ke satu file secara otomatis begitu eksperimen selesai.**
4. Menjalankan analisis statistik dan topic modeling seperti di paper (modul terpisah, dijalankan setelah log terkumpul).

**Bukan tujuan (v1):** UI web, mitigasi drift, dashboard real-time.

---

## 2. Definisi Istilah

| Istilah | Arti |
|---|---|
| **Run** | Satu percakapan penuh (36 tema) antara Agent A dan Agent B + 3 snapshot kuesioner. **1 run = 1 file log.** |
| **Batch** | Kumpulan run dengan konfigurasi sama (mis. 20 run untuk Llama 3.1 70B). |
| **Snapshot** | Titik pengukuran kuesioner: setelah tema 12, 24, 36. |
| **Condition** | Kombinasi model + tipe persona (`none` / `low` / `high`). |
| **RQ1** | Tanpa persona; bandingkan 9 model (efek ukuran & family). |
| **RQ2** | Dengan persona low/high influence pada 2 model yang drift-nya paling besar di RQ1 (di paper: GPT-4o & Llama 3.1 405B). |

---

## 3. Desain Eksperimen (harus mengikuti paper)

### 3.1 Parameter utama

| Parameter | Nilai default (sesuai paper) |
|---|---|
| Jumlah tema | 36 (Aron et al., 1997) |
| Agent per percakapan | 2, masing-masing 1 utterance per tema |
| Urutan giliran per tema | Agent 1 menjawab dulu → Agent 2 menjawab dengan melihat jawaban Agent 1 |
| Temperature percakapan | **0.7** |
| Temperature kuesioner | **0** |
| Pengulangan kuesioner per snapshot | **10×** (untuk mengurangi primacy effect) |
| Snapshot | setelah tema 12, 24, 36 |
| Jumlah run RQ1 | 20 per model |
| Jumlah run RQ2 | 10 per grup persona (low / high), persona **dipasangkan yang mirip** |
| Utterance per run | 36 × 2 = 72 (→ 1.440 utterance per model untuk 20 run) |

Semua nilai di atas **harus bisa diubah lewat file config** (jangan di-hardcode).

### 3.2 Model yang diuji (RQ1)

Sembilan model: GPT-3.5 Turbo, GPT-4o, Llama 3.1 (8B, 70B, 405B), Mixtral (8x7B, 8x22B), Qwen2 (7B, 72B).

Karena akses ke model-model ini beda-beda (dan sebagian sudah deprecated di provider tertentu), tool harus **provider-agnostic**: semua model dipanggil lewat satu interface *OpenAI-compatible chat completions* dengan `base_url`, `api_key_env`, dan `model_name` di config. Dengan begitu bisa dipakai ke OpenAI, OpenRouter, Groq, Together, atau Ollama lokal. Kalau model persis yang dipakai paper tidak tersedia, user mengganti `model_name` di config dan tool mencatatnya di log (lihat §6).

Kategori ukuran (untuk analisis): small (<20B aktif), medium (<100B), large (≥100B). Mixtral memakai *active parameter* (8x7B → 13B, 8x22B → 39B). GPT tidak dimasukkan ke grup ukuran.

### 3.3 Prosedur percakapan

**System prompt percakapan (verbatim dari paper, Appendix B.2):**

```
You are now sharing your thoughts on the question with your partner.
You only reply briefly to your thoughts only for a given question.
```

**Format pesan** — tiap agent memakai role `assistant` untuk ucapannya sendiri dan role `user` untuk tema dan ucapan partner. Contoh:

*Agent 2 menjawab Tema 1:*
```
user:      Question 1 : [Theme 1]
user:      [Jawaban Agent 1 untuk Tema 1]
```

*Agent 1 menjawab Tema 2:*
```
user:      Question 1 : [Theme 1]
assistant: [Jawaban Agent 1, Tema 1]
user:      [Jawaban Agent 2, Tema 1]
user:      Question 2 : [Theme 2]
```

Aturan implementasi:
- Dua pesan `user` berurutan **digabung** jadi satu pesan (dipisah newline) agar kompatibel dengan API yang mewajibkan role bergantian. Pastikan isi dan urutannya tetap sama dengan skema di atas.
- Agent 1 dan Agent 2 punya riwayat pesan **terpisah** (perspektif masing-masing).
- Persona (RQ2) ditambahkan ke system prompt (lihat §3.5).

### 3.4 Snapshot & pengukuran kuesioner

Setelah tema 12, 24, 36 selesai, riwayat percakapan sampai titik itu dimasukkan ke prompt kuesioner (Appendix B.3):

```
System prompt:
Your name is assistant.
Considering the next conversation between user and assistant,
answer given descriptions.
------
[CHATHISTORY]
------
[Questionnaire Setup]
```

- `[CHATHISTORY]` = transkrip percakapan sampai snapshot tersebut.
- `[Questionnaire Setup]` = instruksi skoring dan item kuesioner dari framework **PsychoBench** (Huang et al., 2023).
- Tiap kuesioner dijalankan **10 kali, temperature 0**, per snapshot.
- Diukur untuk **kedua agent** (default). Config `measure_agents: [A, B]` bisa dibatasi ke satu agent untuk menghemat biaya.

**15 kuesioner (4 aspek identitas + MFQ):**

| Aspek | Kuesioner (subskala) |
|---|---|
| Personality | BFI (O, C, E, A, N); EPQ-R (E, P, N, L); DTDD (Machiavellianism, Psychopathy, Narcissism) |
| Interpersonal | BSRI (Masculine, Feminine); CABIN (R, I, A, S, E, C); ICB (Overall); ECR-R (Anxiety, Avoidance); **MFQ-FF** (6 faktor: stimulating companionship, help, intimacy, reliable alliance, self-validation, emotional security) |
| Motivation | GSE; LOT-R; LMS (Rich, Motivator, Important) |
| Emotion | EIS; WLEIS (4 faktor); Empathy |

> ⚠️ **Item kuesioner tidak ada di paper.** Item + scoring PsychoBench harus diambil dari repo PsychoBench (Huang et al., 2023), dan item MFQ dari Mendelson & Aboud (1999). Simpan sebagai file data (`data/questionnaires/*.json`) dengan skema: `name`, `instruction`, `items[]` (teks, reverse-scored?), `subscales{}` (item mana ke subskala mana), `scale_range`, `scoring` (mean/sum). Lisensi dan sitasi harus dicek manual oleh user.

### 3.5 Persona (RQ2)

Dua grup, masing-masing **20 persona**, dibuat dengan data dasar **nama, gender, umur** (meniru use-case chatbot ber-persona):

- **High-influence**: emosional-sensitif, empatik → mudah terpengaruh partner.
- **Low-influence**: *outgoing* dan *goal-oriented*, tidak terkait sensitivitas emosi.

Detail persona ada di Appendix paper (tidak disertakan di PDF ini), jadi tool harus menyediakan:
- `data/personas/high_influence.json` dan `low_influence.json` (diisi/di-generate user; bisa minta LLM membuat 20 persona per grup dari template).
- Aturan pairing: tiap run mengambil **2 persona dari grup yang sama** (yang mirip). 10 run per grup → 10 pasangan per grup.
- Persona digabung ke system prompt: `[persona description]\n\n[system prompt percakapan]`.

### 3.6 Tema (36)

Simpan di `data/themes.json` (urutan 1–36 sesuai Aron et al., 1997; versi di Appendix B.1 paper). Catatan: **Tema 34 di teks PDF terpotong/rancu** (mengulang kalimat Tema 33). Versi asli Aron et al. adalah skenario "rumah Anda terbakar, Anda hanya bisa menyelamatkan satu barang". **Verifikasi manual ke sumber asli sebelum dipakai.**

---

## 4. Fitur & Kebutuhan Fungsional

### FR-1 — Experiment Runner (inti)
- `run` menjalankan 1 batch: N run berurutan (atau paralel terbatas) sesuai config.
- Setiap run: generate 36 tema × 2 agent → snapshot di tema 12/24/36 → simpan log.
- Retry dengan exponential backoff untuk rate limit / error jaringan (maks 5×). Jika tetap gagal, run ditandai `failed` dan tetap menulis log parsial (lihat FR-3).
- Seed run dan `run_id` unik (mis. `{condition}_{timestamp}_{short-uuid}`).

### FR-2 — Questionnaire Engine
- Merender prompt kuesioner persis seperti §3.4, memanggil model 10× (temp 0), mem-parse jawaban jadi skor numerik.
- Parser tahan banting: jika respons tidak bisa di-parse, simpan **raw response**, tandai `parse_ok: false`, coba ulang sekali, lalu lanjut (jangan crash).
- Skoring per subskala (termasuk reverse-scoring) sesuai definisi kuesioner.

### FR-3 — Auto-Save Log (**kebutuhan wajib dari user**)
- **Satu file log per run, dibuat otomatis begitu run selesai** — tanpa perintah manual.
- Lokasi: `logs/{batch_name}/{run_id}.json`.
- Penulisan **atomik**: tulis ke `{run_id}.json.tmp` lalu `rename` ke `.json` hanya setelah run sukses sepenuhnya, supaya file `.json` selalu berarti "run lengkap".
- **Checkpoint**: sepanjang run, state parsial ditulis ke `logs/{batch_name}/{run_id}.partial.json` setiap selesai satu tema/snapshot. Kalau run gagal/di-interrupt (Ctrl+C), file partial dipertahankan dengan `status: "incomplete"`, dan run bisa dilanjutkan dengan `--resume`.
- Selain JSON, tulis juga transkrip mudah dibaca: `{run_id}.transcript.md` (tema → jawaban Agent 1 → jawaban Agent 2).
- Setelah semua run dalam batch selesai, buat `logs/{batch_name}/_batch_summary.json` (daftar run, status, durasi, jumlah error, total token).

### FR-4 — Analisis Statistik (modul terpisah)
Membaca semua log di sebuah folder, lalu per kondisi dan per subskala:
1. Uji normalitas (Shapiro–Wilk) → pilih **repeated-measures ANOVA** (normal) atau **Friedman** (tidak normal) atas 3 snapshot.
2. Post-hoc: **Tukey** (jika ANOVA) atau **Wilcoxon signed-rank + koreksi Bonferroni** (jika Friedman) untuk pasangan snapshot (12–24, 24–36, 12–36).
3. Tandai subskala **"identitas konsisten" (✓)** jika perubahan **tidak signifikan baik di uji omnibus maupun post-hoc** (definisi Table 3 paper).
4. Output: tabel mirip Table 3 (`outputs/table3_{batch}.csv` + `.md`) dengan total ✓ per aspek, plus tabel detail mirip Tabel 10–15 (statistik, Δ antar snapshot, tanda p-value `*`, `**`, `***`).

Library yang dipakai paper: `pandas`, `scipy`, `statsmodels`, `pingouin`.

### FR-5 — Topic Modeling (modul terpisah)
- Unit analisis: **satu utterance** (satu jawaban satu agent untuk satu tema).
- Hanya teks hasil generate; **tema/prompt tidak disertakan**.
- **BERTopic**, hapus stop-words, embedding bahasa Inggris, `min_topic_size=50`.
- Dijalankan terpisah per kondisi (grup ukuran small/medium/large; family GPT/Llama/Mixtral/Qwen; persona low/high).
- Ambil **10 topik teratas** per run analisis → keluarkan kata kunci tiap topik + contoh utterance + (opsional) pemetaan ke nomor tema 1–36.
- Opsional: hitung jumlah kata ganti orang (mis. "I've", "you're") di topic words, seperti analisis pronoun di paper.

### FR-6 — CLI

```
idrift run      --config configs/rq1_llama70b.yaml
idrift run      --config ... --resume logs/rq1_llama70b/
idrift run      --config ... --quick          # 1 run, repeat kuesioner=1 (smoke test)
idrift analyze  --logs logs/rq1_llama70b/ --out outputs/
idrift topics   --logs logs/rq1_*/ --group-by size|family|persona --out outputs/
idrift estimate --config ...                  # estimasi jumlah panggilan API
```

---

## 5. Contoh Config (YAML)

```yaml
batch_name: rq1_llama70b
research_question: RQ1            # RQ1 | RQ2
n_runs: 20
themes_file: data/themes.json

conversation:
  temperature: 0.7
  system_prompt_file: data/prompts/conversation_system.txt
  max_tokens: 512

questionnaire:
  temperature: 0
  repeats: 10
  snapshots: [12, 24, 36]
  measure_agents: [A, B]
  include: all                    # atau daftar: [BFI, EPQ-R, MFQ, ...]

agent_a:
  provider: groq                  # label bebas, hanya untuk log
  base_url: https://api.groq.com/openai/v1
  api_key_env: GROQ_API_KEY
  model: llama-3.1-70b-versatile
  size_group: medium
  persona: null                   # RQ2: path / id persona
agent_b:
  # default: sama dengan agent_a (paper memakai model yang sama untuk kedua agent)
  same_as: agent_a

persona_condition: none           # none | low | high

runtime:
  concurrency: 2
  max_retries: 5
  log_dir: logs
```

> Di paper, kedua agent memakai **model yang sama**. Biarkan `agent_b` bisa berbeda untuk eksperimen lanjutan, tapi default `same_as: agent_a`.

---

## 6. Skema Log (satu file per run)

`logs/{batch_name}/{run_id}.json`

```json
{
  "schema_version": "1.0",
  "run_id": "rq1_llama70b_20261003T101500_ab12",
  "batch_name": "rq1_llama70b",
  "status": "completed",
  "started_at": "...", "finished_at": "...", "duration_sec": 0,
  "condition": {
    "research_question": "RQ1",
    "persona_condition": "none",
    "size_group": "medium",
    "family": "llama3.1"
  },
  "config_snapshot": { "...config lengkap saat run dijalankan..." },
  "models": {
    "agent_a": {"provider": "", "model": "", "persona": null},
    "agent_b": {"provider": "", "model": "", "persona": null}
  },
  "prompts": {
    "conversation_system": "...",
    "questionnaire_system_template": "..."
  },
  "conversation": [
    {
      "theme_id": 1,
      "theme_text": "...",
      "agent_a": {"text": "...", "usage": {"in": 0, "out": 0}, "latency_ms": 0, "finish_reason": "stop"},
      "agent_b": {"text": "...", "usage": {"in": 0, "out": 0}, "latency_ms": 0, "finish_reason": "stop"}
    }
  ],
  "snapshots": [
    {
      "after_theme": 12,
      "questionnaires": [
        {
          "name": "BFI",
          "agent": "A",
          "repeats": [
            {"repeat": 1, "raw_response": "...", "parsed_scores": {"O": 3.4, "C": 3.1}, "parse_ok": true}
          ],
          "mean_scores": {"O": 3.4, "C": 3.1}
        }
      ]
    }
  ],
  "errors": [{"stage": "", "message": "", "retries": 0}],
  "totals": {"api_calls": 0, "tokens_in": 0, "tokens_out": 0}
}
```

Wajib ada: **config snapshot**, **prompt persis yang dikirim**, **nama model aktual**, **raw response kuesioner** (agar parser bisa diperbaiki tanpa mengulang eksperimen), dan **error + retry**.

---

## 7. Struktur Proyek

```
identity-drift/
├── pyproject.toml
├── README.md
├── .env.example
├── configs/                 # satu YAML per batch
├── data/
│   ├── themes.json
│   ├── prompts/
│   ├── questionnaires/      # item + scoring PsychoBench & MFQ (JSON)
│   └── personas/            # high_influence.json, low_influence.json
├── src/idrift/
│   ├── cli.py               # Typer
│   ├── config.py            # Pydantic models + loader YAML
│   ├── llm_client.py        # OpenAI-compatible wrapper, retry, usage tracking
│   ├── conversation.py      # builder pesan & loop 36 tema
│   ├── questionnaire.py     # render prompt, 10× call, parser, scoring
│   ├── runner.py            # orkestrasi run + snapshot + checkpoint
│   ├── logger.py            # tulis JSON atomik, partial, transcript, batch summary
│   ├── analysis/
│   │   ├── stats.py         # normalitas, RM-ANOVA/Friedman, Tukey/Wilcoxon+Bonferroni
│   │   ├── table3.py        # tabel konsistensi ✓
│   │   └── topics.py        # BERTopic
│   └── utils.py
├── tests/
├── logs/                    # output otomatis (di-.gitignore)
└── outputs/                 # tabel & topik hasil analisis
```

**Stack:** Python 3.10+, `openai` (client OpenAI-compatible), `pydantic`, `pyyaml`, `typer`, `tenacity` (retry), `rich` (progress), `pandas`, `scipy`, `statsmodels`, `pingouin`, `bertopic`.

---

## 8. Kebutuhan Non-Fungsional

- **Reprodusibilitas:** semua parameter dari config; config disalin ke log. Paper tidak menyebut seed — catat `seed` jika provider mendukung, tapi jangan diandalkan.
- **Ketahanan:** satu error API tidak boleh menghilangkan data yang sudah terkumpul (checkpoint FR-3).
- **Biaya terkontrol:** `idrift estimate` wajib menampilkan perkiraan panggilan sebelum run dimulai, dan meminta konfirmasi untuk batch besar.
- **Keamanan:** API key hanya dari environment variable / `.env`, **tidak pernah** ditulis ke log atau config snapshot.
- **Idempotent:** run ulang batch yang sama tidak menimpa log lama (pakai `run_id` unik) kecuali `--overwrite`.

### Estimasi beban panggilan API (penting untuk batas free tier)

Per run (default):
- Percakapan: 36 × 2 = **72 panggilan**.
- Kuesioner: 3 snapshot × 15 kuesioner × 10 repeat × 2 agent ≈ **900 panggilan** (kurang lebih; tergantung apakah satu kuesioner = satu prompt berisi semua item, seperti PsychoBench).
- Total ≈ **±970 panggilan/run**; batch 20 run ≈ **±19.400 panggilan**; seluruh RQ1 (9 model) ≈ **±175.000**.

Karena itu sediakan: `--quick` (smoke test), `repeats` yang bisa diturunkan, `measure_agents` satu agent saja, `concurrency` yang bisa diatur, dan rate-limit handling yang baik. Untuk provider gratis, cek batas harian/menit terbaru di dokumentasi masing-masing provider sebelum menjalankan batch penuh; Ollama lokal cocok untuk model kecil (7–8B).

---

## 9. Kriteria Penerimaan (Acceptance Criteria)

1. `idrift run --config configs/example.yaml --quick` menyelesaikan 1 run dan **otomatis membuat** `logs/<batch>/<run_id>.json` + `.transcript.md` tanpa perintah tambahan.
2. File log valid terhadap skema §6 dan memuat 36 entri `conversation` dan 3 entri `snapshots`.
3. Urutan dan role pesan yang dikirim ke API identik dengan skema di §3.3 (ada unit test untuk Agent 1 & 2 pada Tema 1 dan Tema 2).
4. Snapshot dijalankan tepat setelah tema 12, 24, 36 memakai riwayat sampai titik tersebut saja.
5. Kuesioner dipanggil `repeats` kali dengan temperature 0; kegagalan parse tidak menghentikan run.
6. Jika run di-interrupt, `.partial.json` tersisa dan `--resume` melanjutkan tanpa mengulang tema/snapshot yang sudah selesai.
7. `idrift analyze` menghasilkan tabel konsistensi (✓) dan tabel statistik detail dari kumpulan log.
8. `idrift topics` menghasilkan 10 topik teratas per kondisi dengan `min_topic_size=50`.
9. Tidak ada API key di file log mana pun.

---

## 10. Milestone untuk Vibe Coding (urutan prompt yang disarankan)

1. **Skeleton:** setup proyek, `config.py`, `llm_client.py` (satu provider OpenAI-compatible + retry), tes panggilan sederhana.
2. **Conversation engine:** `conversation.py` + unit test format pesan §3.3 dengan *mock LLM* (tanpa API).
3. **Logger:** `logger.py` (JSON atomik, partial, transcript) — uji dengan data palsu.
4. **Runner tanpa kuesioner:** 36 tema, auto-save log. → *Fitur utama user sudah jalan di tahap ini.*
5. **Questionnaire engine:** mulai dari satu kuesioner (BFI), lalu tambah satu per satu sampai 15; tambah snapshot ke runner.
6. **Resume + estimate + concurrency.**
7. **Persona (RQ2):** loader persona, pairing, injeksi ke system prompt.
8. **Analisis statistik** (`stats.py`, `table3.py`).
9. **Topic modeling** (`topics.py`).
10. **README + contoh config** untuk RQ1 dan RQ2.

---

## 11. Asumsi, Ambiguitas, & Risiko

| Hal | Catatan / keputusan default |
|---|---|
| Item PsychoBench & MFQ tidak ada di paper | Harus diambil dari sumber aslinya; tool hanya menyediakan format & loader. |
| Persona detail ada di Appendix yang tidak tersedia | User membuat sendiri 20 persona per grup; deskripsikan dengan kriteria high vs low influence. |
| Tema 34 teks PDF rancu | Verifikasi ke Aron et al. (1997). |
| Agent mana yang dikuesioner? | Paper tidak eksplisit; default **kedua agent**. |
| Unit statistik (apa yang jadi "subjek" di uji repeated-measures) | Paper tidak eksplisit; default: tiap kombinasi (run, agent) = satu subjek, skor snapshot = rata-rata dari 10 repeat. Buat opsi alternatif di config. |
| Model & versi persis | Banyak model sudah berubah/deprecated; hasil tidak akan identik dengan paper. Catat versi model aktual di log. |
| Default temperature provider berubah | Paper memakai 0.7; selalu set eksplisit. |
| Konteks makin panjang saat kuesioner | Pastikan model menampung seluruh riwayat 36 tema (±72 utterance) + item kuesioner; cek context window model kecil. |
| Pesan `user` berurutan | Digabung (lihat §3.3); catat di README sebagai deviasi implementasi minor yang tidak mengubah isi. |

---

## 12. Sitasi

Choi, J., Hong, Y., Kim, M., & Kim, B. (2025). *Examining Identity Drift in Conversations of LLM Agents.* arXiv:2412.00804v2. — Aron et al. (1997) untuk 36 tema; Huang et al. (2023) untuk PsychoBench; Mendelson & Aboud (1999) untuk MFQ; Grootendorst (2022) untuk BERTopic.
