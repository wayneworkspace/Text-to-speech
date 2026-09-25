# Code Review — Video-to-Text Transcription Pipeline

| | |
|---|---|
| **Repo** | `wayneworkspace/Text-to-speech` (nhánh `main`) |
| **Phạm vi review** | `process/main.py`, `process/batch.py`, `process/config.py`, `process/pipeline/{extract,transform,enrich,load}/`, `process/pipeline/{batch_state,utils,gui}.py`, `process/tests/` |
| **Ngày review** | 22/09/2026 |
| **Trạng thái output/** | Trống — chưa có transcript mẫu nào được sinh ra, nên phần 5 bên dưới là **bản mô phỏng (mock-up)**, không phải output thật |

## 1. Tóm tắt tổng quan

Kiến trúc ETL (Extract → Transform → (Diarize + Enrich, optional) → Load) tách bạch rõ ràng, mỗi module chỉ lo một việc, import nặng (`whisper`, `pyannote.audio`, `anthropic`) đều được import trễ (lazy, bên trong hàm) để GUI mở tức thì — đây là một điểm thiết kế tốt, nên giữ nguyên khi mở rộng thêm.

Có **1 vấn đề Critical** cần sửa trước tiên (không liên quan đến logic transcribe, mà là rủi ro lộ secret qua git), và một vài vấn đề Medium/Low về tính đúng đắn và hiệu năng runtime — chi tiết ở mục 2 và 3. Mục 4 là thiết kế đề xuất cho phần bạn hỏi: lưu giọng nói + cấu hình tên để lần sau tự nhận diện — tính năng này **hiện chưa tồn tại** trong code (diarize.py chỉ tách giọng *trong một lần chạy*, nhãn `SPEAKER_00` không có ý nghĩa gì giữa các video khác nhau).

## 2. Đánh giá tính đúng đắn (Correctness)

| Mức độ | Vị trí | Vấn đề | Ảnh hưởng | Đề xuất |
|---|---|---|---|---|
| 🔴 **Critical** | `.env` (root) + `.gitignore` (rỗng) | `.env` đang được **git track** (`git ls-files` xác nhận) và `.gitignore` **rỗng** (0 byte). Đã kiểm tra lịch sử git: `HUGGINGFACE_TOKEN` và `ANTHROPIC_API_KEY` may mắn **luôn rỗng** ở mọi commit trước đó nên chưa có secret nào bị lộ thật — nhưng repo đã có remote `origin = github.com/wayneworkspace/Text-to-speech.git`. Chỉ cần một lần điền token thật vào `.env` rồi `git add . && git commit` là secret sẽ lên GitHub. | Lộ `HUGGINGFACE_TOKEN` / `ANTHROPIC_API_KEY` (tốn phí, bị người khác dùng ké) nếu không sửa trước khi bạn điền giá trị thật. | 1) Thêm `.env`, `__pycache__/`, `output/` vào `.gitignore`. 2) `git rm --cached .env` (giữ file local). 3) Thêm `.env.example` (chỉ có tên key, không có giá trị) để người khác biết cần khai báo gì. 4) Vì chưa từng có giá trị thật bị commit nên **không cần rotate key** — nhưng nên kiểm tra lại trước khi điền token thật vào lần đầu. |
| 🟠 Medium | `extract.py:25` (`probe_media_info`) | `duration = float(info.get("format", {}).get("duration", 0.0))` — nếu ffprobe không trả `format.duration` (một số container mp4 bị remux/thiếu metadata), `duration` âm thầm về `0.0`, không có exception nào được raise. | `plan_chunks()` sẽ trả `[(0.0, 0.0)]`, `cut_chunk` cắt ra đúng 0.05s audio → Whisper transcribe gần như im lặng → file transcript sinh ra gần như trống, **không có lỗi nào hiển thị cho người dùng**, dễ bị tưởng nhầm là "video không có ai nói". | Nếu `duration <= 0`: raise `RuntimeError` rõ ràng ngay tại `probe_media_info`, hoặc fallback đọc duration từ `streams[].duration` trước khi bỏ cuộc. |
| 🟡 Low | `transform.py:30-62` (`plan_chunks`) / `config.py` | Không có validation cho `CHUNK_MIN_SECONDS < CHUNK_TARGET_SECONDS < CHUNK_MAX_SECONDS` hay việc các giá trị này phải dương. Nếu người dùng gõ nhầm trong `.env` (vd. min > max), thuật toán vẫn chạy nhưng cửa sổ tìm điểm cắt im lặng `[window_start, window_end]` có thể rỗng, khiến toàn bộ chunk bị cắt cứng tại `ideal_end` — không crash, nhưng hành vi im lặng, khó debug. | Chunk size không như mong đợi, không có cảnh báo. | Validate thứ tự + tính dương của 3 giá trị này ngay trong `config.py` khi load, `raise` một lỗi rõ ràng nếu sai. |
| 🟡 Low | `extract.py:20,44`, `transform.py:24,76` | Tất cả các lệnh `subprocess.run(cmd, ...)` gọi `ffmpeg`/`ffprobe` đều **không có `timeout=`**. | Nếu gặp file input hỏng/stream mạng bị treo, pipeline có thể treo vô thời hạn mà không có cách nào báo lỗi cho người dùng (GUI chỉ thấy thanh progress chạy mãi). | Thêm `timeout=` hợp lý (vd. vài phút tuỳ độ dài video) và bắt `subprocess.TimeoutExpired` để log lỗi rõ ràng thay vì treo. |
| ℹ️ Info | `main.py:34` và `main.py:92` | `check_ffmpeg_available()` được gọi 2 lần (trong `main()` trước khi mở GUI, và lại trong `run_pipeline()`). Vô hại (chỉ là `shutil.which`, rất rẻ) nhưng dư thừa. | Không đáng kể. | Có thể bỏ 1 trong 2 lần gọi nếu muốn code gọn hơn — không bắt buộc. |

Các phần còn lại của pipeline (silence detection, chunk cutting bằng `-ss` trước `-i` để seek nhanh trên PCM, error handling try/finally dọn temp dir, threading giữa GUI và pipeline qua `queue.Queue`) đều được viết đúng và cẩn thận — không có vấn đề đáng kể.

## 3. Đánh giá tối ưu thời gian chạy (Performance)

| Mức độ | Vị trí | Vấn đề | Ảnh hưởng | Đề xuất |
|---|---|---|---|---|
| 🔴 High | `main.py:54` → `transform.py:81-87` (`load_whisper_model`) | Model Whisper (`medium` ≈ 1.5GB, `large-v3` còn nặng hơn) được **load lại từ đầu mỗi lần `run_pipeline()` chạy** — kể cả khi người dùng xử lý nhiều video liên tiếp trong cùng một phiên GUI (nút "Choose MP4 video..." được bật lại sau mỗi lần chạy xong, cho phép chọn video tiếp theo). | Với video thứ 2, 3... người dùng phải chờ thêm hàng chục giây đến vài phút load lại model **không cần thiết**, dù model size không đổi giữa các lần chạy. | Cache model đã load ở cấp `App`/module (key theo `model_size`), chỉ load lại nếu `MODEL_SIZE` trong config thay đổi. Đơn giản nhất: load model 1 lần trong `main()` trước khi mở GUI, truyền vào `run_pipeline` qua closure/partial. |
| 🟠 Medium | `transform.py:81-87` | `whisper.load_model(model_size)` không truyền `device` hay `fp16` tường minh, và không log thiết bị đang dùng (CPU hay GPU). | Người dùng không biết vì sao chạy chậm (đang chạy CPU hay GPU?), không có cách ép CPU nếu VRAM không đủ; trên CPU, thư viện Whisper sẽ tự in cảnh báo "FP16 is not supported on CPU" **mỗi lần transcribe 1 chunk** vì `fp16=True` là default. | `device = "cuda" if torch.cuda.is_available() else "cpu"`, log ra (`log(f"Using device: {device}")`), truyền `device=device` vào `load_model` và `fp16=(device=="cuda")` vào `transcribe_chunk` để tắt cảnh báo thừa và chủ động kiểm soát độ chính xác số học. |
| 🟠 Medium | `main.py:60-71` | Diarization (pyannote, `diarize.py`) và transcription (Whisper, `transform.py`) chạy **tuần tự**, dù cả hai chỉ phụ thuộc vào cùng file `audio.wav` đã extract — không phụ thuộc lẫn nhau. | Tổng thời gian chạy = thời gian Whisper + thời gian pyannote, thay vì `max()` của hai cái nếu chạy song song. Trên máy có GPU + đủ VRAM, hoặc máy nhiều core, đây là cơ hội giảm runtime thực sự đáng kể cho video dài. | Cân nhắc chạy 2 tác vụ này song song bằng `concurrent.futures.ThreadPoolExecutor` (cẩn thận tranh chấp VRAM nếu cả hai cùng dùng GPU — có thể ép pyannote chạy CPU nếu Whisper đang dùng GPU). Đây là optimization nên làm **sau** khi đã có cache model (mục High ở trên), lợi ích/độ phức tạp thấp hơn. |
| 🟡 Low | `diarize.py:44-59` (`assign_speakers`) | Vòng lặp lồng O(số segment × số turn) để tìm turn overlap nhiều nhất cho mỗi segment. | Với cuộc họp thông thường (vài chục turn, vài trăm segment) không đáng kể; với recording nhiều giờ, nhiều người nói, có thể chậm dần. | Không cấp thiết. Nếu cần scale: sort `turns` theo `t_start`, dùng `bisect` để giới hạn phạm vi tìm kiếm thay vì quét toàn bộ. |
| 🟡 Low | `transform.py:128-148` (`transform_to_segments`) | Cắt chunk (ffmpeg, I/O-bound) và transcribe chunk (Whisper, CPU/GPU-bound) chạy tuần tự từng chunk một, không "prefetch" chunk tiếp theo trong lúc chunk hiện tại đang transcribe. | Nhỏ — vì cắt bằng `-c copy` (stream copy trên PCM) rất nhanh so với thời gian Whisper infer, nên overlap sẽ tiết kiệm không nhiều. | Không ưu tiên; chỉ đáng làm nếu đã tối ưu hết các mục trên. |
| 💡 Gợi ý thêm | `requirements.txt` | Đang dùng `openai-whisper` (PyTorch thuần). | — | Nếu cần tốc độ hơn nữa: cân nhắc `faster-whisper` (backend CTranslate2), thường nhanh hơn 2-4 lần và tốn ít RAM/VRAM hơn ở cùng model size, API tương tự nên chi phí migrate không quá lớn. Không bắt buộc, chỉ nêu để tham khảo khi pipeline cần xử lý khối lượng lớn. |

## 4. Giải pháp: lưu giọng nói người nói + tự nhận diện ở lần chạy sau

### 4.1 Vì sao chưa làm được hiện tại

`pipeline/diarize.py` dùng `pyannote/speaker-diarization-community-1` — đây là **diarization** (phân cụm giọng nói *trong một file audio*), không phải **speaker identification**. Nhãn `SPEAKER_00`, `SPEAKER_01`... chỉ là ID cụm tạm thời cho *lần chạy đó*; chạy lại chính video đó hay chạy video khác, `SPEAKER_00` không đảm bảo là cùng một người. Đã kiểm tra tài liệu chính thức của `pyannote/speaker-diarization-community-1` — pipeline này **không có tham số trả về embedding giọng nói** kèm kết quả diarization, nên không thể "tận dụng luôn" pipeline hiện tại; cần thêm một lớp **speaker identification** riêng, đặt lên trên diarization.

### 4.2 Kiến trúc đề xuất (3 bước)

1. **Enrollment (một lần, khi có người nói mới)** — trích embedding giọng nói (vector đặc trưng cố định chiều) từ một đoạn audio sạch của người đó, gán tên, lưu vào một "kho hồ sơ giọng nói" cục bộ.
2. **Diarization (mỗi lần chạy, như hiện tại)** — `pyannote/speaker-diarization-community-1` tách các turn theo cụm `SPEAKER_XX` như code hiện có.
3. **Matching (bước mới, chạy sau diarization, trước khi ghi markdown)** — với mỗi cụm `SPEAKER_XX` trong video hiện tại, tính embedding đại diện (centroid) từ các turn dài nhất của cụm đó, so cosine similarity với từng hồ sơ đã enroll; nếu vượt ngưỡng → gán tên thật; nếu không có hồ sơ nào khớp → gán nhãn tuần tự "Speaker 1", "Speaker 2"... theo thứ tự xuất hiện lần đầu trong video (thay vì để lộ `SPEAKER_00` thô).

Đây đúng là quy trình được khuyến nghị trong thảo luận chính thức của pyannote về "nhận diện người nói đã biết" (xem Sources cuối file).

### 4.3 Model embedding đề xuất

`speaker-diarization-community-1` không tự lộ embedding, nên cần thêm một model embedding riêng, dùng `Model.from_pretrained(...)` + `Inference(model, window="whole")` để ra 1 vector duy nhất cho mỗi đoạn audio. Hai lựa chọn phổ biến, đều dùng chung `HUGGINGFACE_TOKEN` đã có sẵn trong `.env`:

- **`pyannote/wespeaker-voxceleb-resnet34-LM`** — chính là model embedding mà các pipeline diarization của pyannote dùng nội bộ, nên tương thích tốt với hệ sinh thái đang dùng.
- **`speechbrain/spkrec-ecapa-voxceleb`** — ECAPA-TDNN, cũng rất phổ biến cho speaker verification, thường không bị gate (không cần accept license riêng).

Cả hai đều cho ra vector vài trăm chiều, so sánh bằng cosine similarity (`scipy.spatial.distance.cosine` hoặc dot product trên vector đã normalize).

### 4.4 Kho lưu hồ sơ giọng nói (Speaker Profile Store)

File mới `speaker_profiles.json` ở root project (**phải thêm vào `.gitignore`** — dữ liệu voiceprint là dữ liệu cá nhân/sinh trắc học nhạy cảm, không nên đẩy lên git dùng chung, kể cả private repo):

```json
{
  "speakers": [
    {
      "name": "Nguyễn Văn A",
      "embedding": [0.0123, -0.0456, "... (192 hoặc 256 chiều tuỳ model)"],
      "sample_count": 3,
      "updated_at": "2026-09-22T10:00:00+07:00"
    },
    {
      "name": "Trần Thị B",
      "embedding": ["..."],
      "sample_count": 1,
      "updated_at": "2026-09-15T09:00:00+07:00"
    }
  ]
}
```

`embedding` nên là **centroid** (trung bình) của nhiều lần enroll cho cùng một người — càng nhiều mẫu, càng ổn định trước nhiễu/âm lượng khác nhau giữa các lần ghi âm.

### 4.5 Module mới: `pipeline/speaker_id.py`

Không đổi cấu trúc hiện có — thêm 1 module mới cùng cấp `diarize.py`, `enrich.py`, giữ đúng nguyên tắc "mỗi module một việc" của repo:

```python
def extract_embedding(audio_path: str, start: float, end: float) -> np.ndarray: ...
def compute_speaker_centroids(audio_path: str, turns: list) -> dict[str, np.ndarray]: ...
def load_profiles(path: str) -> list[dict]: ...
def save_profiles(path: str, profiles: list[dict]): ...
def match_speakers(centroids: dict, profiles: list[dict], threshold: float) -> dict[str, str]: ...
def enroll_speaker(name: str, audio_path: str, turns: list, path: str): ...
```

Gọi trong `main.py`, ngay sau `assign_speakers(segments, turns)` hiện có (dòng 62) và trước bước `enrich.py`:

```python
if CONFIG["huggingface_token"]:
    turns = diarize_audio(audio_path, CONFIG["huggingface_token"], log)
    segments = assign_speakers(segments, turns)

    profiles = load_profiles(CONFIG["speaker_profiles_path"])
    centroids = compute_speaker_centroids(audio_path, turns)
    name_map = match_speakers(centroids, profiles, CONFIG["speaker_match_threshold"])
    segments = resolve_speaker_names(segments, name_map)   # SPEAKER_00 -> "Nguyễn Văn A" hoặc "Speaker 1"
```

### 4.6 Cấu hình mới (`.env` / `config.py`)

| Key | Mặc định | Ý nghĩa |
|---|---|---|
| `SPEAKER_PROFILES_PATH` | `speaker_profiles.json` (root) | Đường dẫn kho hồ sơ giọng nói. |
| `SPEAKER_MATCH_THRESHOLD` | `0.75` | Ngưỡng cosine similarity để coi là "cùng một người". Cần tune thực tế trên dữ liệu của bạn — thấp hơn = dễ nhận nhầm người khác, cao hơn = dễ bỏ sót (rơi về "Speaker N"). |
| `AUTO_UPDATE_SPEAKER_PROFILES` | `false` | Nếu `true`: mỗi khi match thành công với similarity cao, tự cập nhật (trung bình thêm) embedding của người đó để hồ sơ ngày càng chính xác hơn. Nên để `false` ban đầu để tránh "trôi" hồ sơ do nhận nhầm. |

### 4.7 Logic đặt tên hiển thị cuối cùng (đúng theo yêu cầu output của bạn)

Với mỗi nhãn `SPEAKER_XX` xuất hiện trong video, theo **thứ tự xuất hiện lần đầu**:

- Nếu `match_speakers()` tìm được hồ sơ khớp (similarity ≥ ngưỡng) → dùng **tên thật** đã cấu hình.
- Nếu không khớp với hồ sơ nào → gán **"Speaker 1"**, **"Speaker 2"**... theo thứ tự xuất hiện (không dùng nhãn thô `SPEAKER_00` từ pyannote, vì nó không thân thiện với người đọc).

Đây chính là hành vi bạn mô tả trong "đầu ra mong muốn" — chỉ cần thêm bước này, `load.py` (phần ghi markdown, đã hỗ trợ sẵn timestamp + prefix tên người nói ở dòng 53) **không cần sửa gì thêm**.

### 4.8 Trải nghiệm enroll (gợi ý, không bắt buộc làm ngay)

- **Cách nhanh nhất, không cần ghi âm riêng**: sau khi 1 video đã chạy xong và có diarization, thêm một nút nhỏ trong GUI hiện có (`pipeline/gui.py`) — "Gán tên cho giọng nói" — liệt kê các `SPEAKER_XX` tìm được, cho người dùng gõ tên tương ứng, rồi tính centroid từ chính các turn của video đó và lưu vào `speaker_profiles.json`. Không tốn thêm bước ghi âm.
- **Cách chuẩn hơn**: script CLI riêng `process/enroll_speaker.py <file_audio_sach.wav> "Tên người nói"` để enroll từ một đoạn ghi âm sạch, ngắn (khuyến nghị 20-60 giây, ít nhiễu, một người nói duy nhất).

### 4.9 Lưu ý về quyền riêng tư dữ liệu giọng nói

Vì đây là review cho ngữ cảnh công ty Data: embedding giọng nói là dữ liệu mang tính sinh trắc học (biometric-adjacent). Khuyến nghị: (1) không commit `speaker_profiles.json` lên git dùng chung — đã nêu ở 4.4; (2) nên có cơ chế xoá hồ sơ theo tên khi người đó yêu cầu; (3) nếu dùng trong môi trường nhiều người truy cập chung máy, cân nhắc quyền đọc/ghi file `speaker_profiles.json` hoặc mã hoá tại chỗ nếu công ty có chính sách compliance liên quan đến dữ liệu sinh trắc học.

## 5. Mẫu minh hoạ output markdown mong muốn

`output/` hiện đang trống (chưa có video nào được xử lý), nên đây là **bản mô phỏng** minh hoạ định dạng đầu ra mong muốn sau khi triển khai mục 4 — không phải kết quả chạy thật. `load.py` hiện tại đã render đúng timestamp + prefix tên (dòng 52-53); phần còn thiếu duy nhất là bước "map SPEAKER_XX → tên thật hoặc Speaker N" ở mục 4.7 trước khi gọi `write_markdown()`.

```markdown
# Transcript: cuoc-hop-du-an-Q3.mp4

> ⚠️ marks a line Whisper was not confident about - worth a manual check.

## Mở đầu cuộc họp

**[00:00] Nguyễn Văn A:** Chào mọi người, hôm nay chúng ta họp về tiến độ dự án Q3.

**[00:12] Speaker 2:** Vâng, em xin phép báo cáo phần backend trước ạ.

**[00:47] Nguyễn Văn A:** Được, cảm ơn em.

## Thảo luận ngân sách Q3

**[05:03] Trần Thị B:** Về phần ngân sách, hiện mình đang vượt khoảng 8% so với kế hoạch. ⚠️

**[05:31] Speaker 2:** Em nghĩ mình nên rà lại chi phí bên vendor trước.
```

Trong ví dụ trên: "Nguyễn Văn A" và "Trần Thị B" là hai giọng đã được enroll từ trước và khớp hồ sơ (similarity ≥ ngưỡng); "Speaker 2" là một giọng diarization tách ra được nhưng không khớp bất kỳ hồ sơ nào trong `speaker_profiles.json` — đúng theo yêu cầu, được gán nhãn tuần tự thay vì để trống hoặc hiện `SPEAKER_01` thô.

## 6. Đề xuất thứ tự triển khai

1. **Critical trước** — thêm `.gitignore`, `git rm --cached .env`, thêm `.env.example` (mục 2).
2. **Quick win hiệu năng** — cache Whisper model giữa các lần chạy trong cùng phiên GUI, log rõ CPU/GPU đang dùng (mục 3, hàng High/Medium đầu).
3. **Vá 2 lỗi đúng đắn còn lại** — fallback khi ffprobe thiếu duration, validate thứ tự chunk min/target/max (mục 2).
4. **Tính năng chính bạn yêu cầu** — module `speaker_id.py` + `speaker_profiles.json` + logic đặt tên/Speaker N (mục 4) — đây là phần việc lớn nhất.
5. *(Tuỳ chọn, không cấp thiết)* — chạy song song diarization + transcription, cân nhắc `faster-whisper`.

Nếu bạn muốn, mình có thể bắt đầu triển khai trực tiếp vào source code theo đúng thứ tự trên — chỉ cần xác nhận là bắt đầu từ mục nào.

---

### Sources (tham khảo cho mục 4)

- [pyannote/speaker-diarization-community-1 — Hugging Face model card](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [Community-1: Unleashing open-source diarization — pyannoteAI blog](https://www.pyannote.ai/blog/community-1)
- [Speaker Diarization of Known Speakers — pyannote/pyannote-audio Discussion #1667](https://github.com/pyannote/pyannote-audio/discussions/1667)
- [pyannote/wespeaker-voxceleb-resnet34-LM — Hugging Face](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)


---

## 7. Phụ lục: Kiến trúc thư mục cho batch xử lý 80 file (1 máy, GPU NVIDIA)

Bối cảnh: 80 file video, ~5GB/file (~400GB tổng), chạy trên 1 máy có GPU NVIDIA. Nguyên tắc: xử lý tuần tự qua GPU đơn (an toàn VRAM, tránh out-of-memory), có khả năng **resume** nếu bị gián đoạn giữa chừng (batch cả trăm GB có thể chạy nhiều giờ đến vài ngày), và **không đổi logic pipeline/ hiện có** — chỉ thêm lớp điều phối batch bên ngoài.

```
project/
├── README.md
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
│
├── data/
│   └── speaker_profiles.json        # kho voice embedding đã enroll — DÙNG CHUNG xuyên suốt cả 80 file
│
├── output/
│   └── [dd-mm-yy] - <video_name> - transcript.md   # 1 markdown/video, như hiện tại — không đổi
│
├── logs/
│   ├── batch_run_[timestamp].log     # log tổng của cả lần chạy batch
│   └── per_file/<video_name>.log     # log riêng từng video — dò lỗi 1 file mà không phải đọc log của cả 80 file
│
├── state/
│   └── batch_state.json              # trạng thái từng file: pending / running / done / failed + output_path
│
└── process/
    ├── main.py                       # giữ nguyên — GUI, xử lý 1 file, tương tác
    ├── batch.py                      # MỚI — CLI, quét INPUT_DIR, xử lý N file tuần tự, đọc/ghi state
    ├── config.py                     # thêm INPUT_DIR, STATE_PATH, LOG_DIR, SPEAKER_PROFILES_PATH
    └── pipeline/
        ├── extract.py / transform.py / diarize.py / enrich.py / load.py / utils.py   # không đổi
        ├── speaker_id.py             # MỚI — enrollment + nhận diện giọng nói (mục 4 ở trên)
        └── batch_state.py            # MỚI — helper đọc/ghi state/batch_state.json
```

**80 file gốc (400GB) không cần copy vào project** — chỉ cần `INPUT_DIR=` trong `.env` trỏ tới ổ đĩa hiện có chứa chúng; `batch.py` chỉ đọc, không di chuyển/copy file gốc.

Vài điểm then chốt:

- `state/batch_state.json` là phần quan trọng nhất ở quy mô 80 file: nếu batch bị dừng giữa chừng (mất điện, Ctrl+C, 1 file lỗi khiến ffmpeg/whisper crash), lần chạy lại chỉ xử lý các file `pending`/`failed`, bỏ qua file đã `done` — không chạy lại từ đầu.
- `logs/per_file/` tách riêng vì batch chạy không có người giám sát liên tục — cần biết chính xác file nào lỗi và lỗi gì mà không làm dừng cả batch (giữ nguyên try/except hiện có trong `main.py`, chỉ log + đánh dấu `failed` rồi chuyển qua file tiếp theo).
- `data/speaker_profiles.json` tách khỏi `output/` vì đây là dữ liệu "sống" được đọc/cập nhật liên tục qua cả 80 file, không phải một sản phẩm output như transcript.
- Với 1 GPU: `batch.py` nên mặc định chạy **tuần tự** (1 video/lần) ở bước Whisper + diarization để tránh tràn VRAM, thay vì cố chạy song song nhiều video trên cùng 1 GPU.


---

## 8. Tổng hợp: checklist đề xuất bổ sung cho source code

Toàn bộ nội dung dưới đây là các đề xuất đã trao đổi trong quá trình review — **chưa có dòng code nào được sửa**, đây là bản tổng hợp để bạn quyết định làm phần nào trước. Đánh dấu (❌ chưa làm) cho tất cả.

### 8.1 Tự động hoá xử lý video mới (bổ sung — chưa có ở mục 4/7 phía trên)

Vì video mới xuất hiện không đều đặn (không theo lịch cố định):

- ❌ Windows Task Scheduler chạy `process/batch.py` **định kỳ mỗi 30-60 phút** (không phải cron 1 lần/ngày — độ trễ 24h không phù hợp khi video đến bất kỳ lúc nào). Không cần n8n/Airflow — bài toán chỉ là 1 pipeline tuyến tính trên 1 máy/1 GPU, các công cụ đó giải quyết vấn đề khác (nối nhiều dịch vụ ngoài, điều phối nhiều pipeline song song trên nhiều máy).
- ❌ Cơ chế **khoá (lock file)**, ví dụ `state/.batch.lock` — `batch.py` tự kiểm tra và thoát ngay nếu phát hiện đã có 1 tiến trình khác đang chạy, tránh 2 lần chạy chồng lên nhau tranh giành cùng 1 GPU.
- ❌ Kiểm tra **kích thước file ổn định** (đọc size 2 lần cách nhau vài giây) trước khi xử lý — tránh bắt phải video đang được copy dở dang vào `INPUT_DIR`, gây lỗi ffprobe/ffmpeg giữa chừng.
- ❌ *(Tuỳ chọn, chỉ cần nếu độ trễ vài chục phút không chấp nhận được)* dùng thư viện `watchdog` để theo dõi `INPUT_DIR` theo thời gian thực thay vì poll định kỳ — nặng hơn, cần chạy như 1 service nền liên tục, không bắt buộc ở quy mô hiện tại.

### 8.2 Checklist tổng hợp toàn bộ (theo nhóm, xem chi tiết ở mục tương ứng)

**Sửa lỗi & bảo mật (mục 2)**
- ✅ `.gitignore` (thêm `.env`, `__pycache__/`, `output/`) + `git rm --cached .env` + `.env.example` — đã làm, `git rm --cached .env` đã stage (chưa commit)
- ✅ `extract.py`: raise lỗi rõ ràng khi ffprobe không trả `duration` (thay vì âm thầm về 0.0)
- ✅ `config.py`: validate `CHUNK_MIN/TARGET/MAX_SECONDS` (thứ tự + dương)
- ✅ `extract.py` / `transform.py`: thêm `timeout=` cho các lệnh `subprocess.run` gọi ffmpeg/ffprobe

**Tối ưu runtime (mục 3)**
- ✅ Cache model Whisper giữa các lần chạy trong cùng phiên (không load lại mỗi video)
- ✅ Log rõ CPU/GPU đang dùng, truyền `device`/`fp16` tường minh vào Whisper
- ❌ *(tuỳ chọn)* chạy song song diarization + transcription
- ❌ *(tham khảo)* cân nhắc `faster-whisper` nếu cần nhanh hơn nữa

**Nhận diện giọng nói (mục 4)**
- ✅ `pipeline/speaker_id.py` (mới): extract embedding, tính centroid, load/save/match hồ sơ
- ✅ `speaker_profiles.json` (thêm vào `.gitignore`) — kho hồ sơ giọng nói đã enroll
- ✅ `config.py` + `.env`: `SPEAKER_PROFILES_PATH`, `SPEAKER_MATCH_THRESHOLD`, `AUTO_UPDATE_SPEAKER_PROFILES`
- ✅ Logic gán tên cuối: khớp hồ sơ → tên thật; không khớp → "Speaker 1", "Speaker 2"... (không phải `SPEAKER_00` thô)
- ✅ *(một phần)* script CLI `process/enroll_speaker.py` đã có — nút trong GUI thì chưa làm

**Batch xử lý nhiều file (mục 7)**
- ✅ `process/batch.py` (mới) — CLI quét `INPUT_DIR` đệ quy, chạy tuần tự từng video qua `main.py`'s `run_pipeline()`
- ✅ `pipeline/batch_state.py` (mới) + `state/batch_state.json` — resume được nếu bị gián đoạn, kèm lock file có phát hiện "lock chết" tự động (mục 9.1)
- ✅ `logs/batch_run_*.log` + `logs/per_file/<video>.log`
- ✅ `config.py`: thêm `INPUT_DIR`, `STATE_PATH`, `LOCK_PATH`, `LOG_DIR`, `EXTRACT_CHUNK_SECONDS`, `EXTRACT_MAX_WORKERS`

**Tự động hoá (mục 8.1 — mới thêm)**
- ✅ `batch.py` sẵn sàng cho Task Scheduler chạy định kỳ (30-60 phút) — lock file + kiểm tra kích thước file ổn định trước khi xử lý, đều đã code + test (mục 9.1)

**Tổ chức lại code & tăng tốc trích xuất (yêu cầu bổ sung 22/09/2026)**
- ✅ `pipeline/` tách thành 4 thư mục con theo đúng 4 giai đoạn ETL ở mục 1 (`extract/`, `transform/`, `enrich/`, `load/`) - dễ đọc hơn cho cả intern lẫn senior; không dùng `__init__.py` (project target Python 3.13, dùng namespace package - PEP 420), import trực tiếp từ file con (mục 9.2)
- ✅ `extract_audio()`: trích xuất song song theo chunking thời gian (`ThreadPoolExecutor`) cho video dài, tự fallback về single-pass cho video ngắn (mục 9.2)
- ✅ `process/tests/` — bộ test `unittest`, nhóm `unit/` (logic thuần, 64 test) + `integration/` (ffmpeg thật, 8 test), toàn bộ 72 test **pass** (mục 9.3)

**Đã cân nhắc nhưng tạm hoãn (theo yêu cầu của bạn)**
- ⏸ Structured content output (JSON lines/Parquet/DB, mục 6-7 trao đổi) — hoãn lại cho tới khi có nhu cầu lọc/tổng hợp/join cụ thể, hiện tại markdown + grep là đủ.


---

## 9. Trạng thái triển khai (cập nhật 22/09/2026)

**Phase 1 + speaker_id.py — đã code + tự kiểm tra được:** `.gitignore`/`.env.example`/gỡ `.env` khỏi git (đã `git rm --cached`, **chưa commit** — bạn xem lại rồi tự commit, hoặc báo mình commit giúp), sửa lỗi ffprobe duration, validate chunk config, timeout cho ffmpeg/ffprobe, cache model Whisper + log device/fp16, `pipeline/speaker_id.py`, `process/enroll_speaker.py`, wiring vào `main.py`, cập nhật README.md.

**Vòng cập nhật thứ 2 (cùng ngày, theo yêu cầu tiếp theo của bạn)** — 4 việc: (1) sửa 2 lỗ hổng batch runner theo đề xuất đã thống nhất, (2) nhóm lại `pipeline/` theo giai đoạn ETL cho dễ đọc, (3) tạo bộ test có tổ chức nhóm, (4) tăng tốc trích xuất audio bằng chunking song song. Chi tiết từng phần ở 9.1-9.3 dưới đây.

### 9.1 Batch runner: 2 lỗ hổng đã sửa (lock file "chết" + tiến trình bị ngắt giữa chừng)

Bạn hỏi liệu batch runner có xử lý được (a) 2 lần chạy cron chồng lên nhau cùng xử lý trùng 1 video, và (b) 1 lần chạy bị ngắt giữa chừng (crash/mất điện) thì lần sau chạy lại có sao không. Bản phác thảo ban đầu ở mục 7-8 **chưa** xử lý an toàn 2 trường hợp này: một lock file chỉ kiểm tra "file có tồn tại" sẽ tự khoá chết vĩnh viễn nếu tiến trình giữ lock crash mà không kịp xoá lock khi thoát (đúng loại lỗi dự án này vừa gặp thật với `.git/index.lock` trong chính phiên review này); và một video đang `"running"` khi bị ngắt sẽ mắc kẹt ở trạng thái đó mãi, không bao giờ được thử lại.

Đã sửa trong `pipeline/batch_state.py`:

- **`acquire_lock()`** — khi lock file đã tồn tại, kiểm tra PID ghi trong đó còn sống không (`psutil.pid_exists`) **và** tuổi lock có vượt ngưỡng an toàn không (`_MAX_LOCK_AGE_SECONDS`, mặc định 6 giờ — lưới an toàn thứ 2 cho trường hợp tiến trình treo chứ không chết hẳn). PID đã chết **hoặc** lock quá cũ → coi là lock rác từ lần crash trước, tự xoá và log rõ ràng, rồi tiếp tục chạy thay vì thoát im lặng.
- **`recover_interrupted()`** — gọi ngay sau khi `acquire_lock()` thành công: vì tại một thời điểm chỉ duy nhất 1 tiến trình giữ được lock, bất kỳ video nào còn ghi `"running"` trong `state/batch_state.json` lúc này chắc chắn là tàn dư của lần chạy trước bị crash giữa chừng — được reset về `"pending"` (kèm tăng `retry_count`) để thử lại, thay vì kẹt vĩnh viễn.
- **`MAX_RETRIES`** (mặc định 3) — một video lỗi liên tục qua nhiều lần cron cũng cần có điểm dừng, nếu không sẽ tốn 1 lượt retry mỗi 30-60 phút mãi mãi cho 1 file hỏng vĩnh viễn. Sau 3 lần thất bại, `should_process()` tự chuyển video đó sang `"failed"` cố định và log rõ để bạn vào `logs/` kiểm tra tay.

Toàn bộ logic trên đã có unit test thật (spawn rồi kill 1 tiến trình con để giả lập PID chết, giả lập lock quá cũ, giả lập lock file hỏng/không đọc được JSON, giả lập video `"running"` bị bỏ dở...) — xem 9.3.

### 9.2 Nhóm lại `pipeline/` theo giai đoạn ETL + tăng tốc trích xuất bằng chunking

**Nhóm lại thư mục** — `pipeline/extract.py`, `transform.py`, `diarize.py`, `speaker_id.py`, `enrich.py`, `load.py` (6 file phẳng cùng cấp) được tách thành 4 thư mục con đúng 4 giai đoạn ở mục 1 (Kiến trúc): `pipeline/extract/`, `pipeline/transform/`, `pipeline/enrich/` (gộp `diarize.py` + `speaker_id.py` + `enrich.py` — cả 3 đều là bước "làm giàu" tuỳ chọn, chạy sau transform), `pipeline/load/`. `pipeline/gui.py` và `pipeline/utils.py` giữ nguyên ở gốc `pipeline/` vì không thuộc riêng giai đoạn nào (cross-cutting). Đã kiểm tra `git status`: đây là các phép move sạch (file cũ hiện `D` - deleted, file mới hiện `??` - untracked), không có code cũ nào bị bỏ sót hay nhân bản.

Bản đầu của việc nhóm lại này có tạo `__init__.py` re-export cho mỗi thư mục con (để `main.py` import gọn 1 dòng/giai đoạn). Theo yêu cầu sau đó của bạn (dự án target Python 3.13), toàn bộ `__init__.py` đã được xoá — từ Python 3.3 (PEP 420), một thư mục không có `__init__.py` vẫn là "namespace package" import được bình thường, nên không bắt buộc phải có file này nữa. Đã cập nhật `main.py`/`enroll_speaker.py` để import thẳng từ file con thay vì qua re-export, ví dụ `from pipeline.extract.extract import probe_media_info, extract_audio` thay vì `from pipeline.extract import probe_media_info, extract_audio`. Đã test lại: `py_compile` + import thật toàn bộ module liên quan đều pass, 72 test vẫn pass nguyên (mục 9.3).

**Lưu ý riêng cho `process/tests/`**: xoá `__init__.py` ở đây có một tác dụng phụ đã kiểm chứng thật (kể cả trên Python 3.13, không chỉ 3.10 của máy review) — `unittest`'s CLI `discover` chấp nhận thư mục bắt đầu (`-s tests`) là namespace package, nhưng **không tự đệ quy vào thư mục con** cũng là namespace package (`tests/unit/`, `tests/integration/`) để tìm test bên trong — nên lệnh gộp `discover -s tests` sẽ lặng lẽ báo "Ran 0 tests" dù code hoàn toàn đúng. Discover trực tiếp vào từng nhóm (`discover -s tests/unit`, `discover -s tests/integration`) thì vẫn chạy bình thường. Đã thêm `process/tests/run_all.py` để có lại 1 lệnh chạy gộp cả 2 nhóm (gọi `TestLoader().discover()` trực tiếp cho từng thư mục thay vì dựa vào CLI đệ quy) — xem `process/tests/README.md`.

**Trích xuất song song theo chunking** — `extract_audio()` nhận thêm `chunk_seconds` (mặc định 600s = 10 phút, cấu hình qua `EXTRACT_CHUNK_SECONDS`) và `max_workers` (mặc định 4, qua `EXTRACT_MAX_WORKERS`). Video dài hơn ngưỡng này được chia thành các đoạn thời gian `chunk_seconds`, chạy `ffmpeg` trên từng đoạn song song bằng `ThreadPoolExecutor` (tận dụng nhiều core CPU đang rảnh trong lúc GPU bận transcribe video *khác*), rồi ghép lại bằng ffmpeg concat demuxer. Video ngắn hơn ngưỡng, hoặc không xác định được duration, luôn dùng single-pass như cũ. Trade-off: mỗi đoạn cắt bằng seek nhanh (`-ss` trước `-i`), cùng kỹ thuật `cut_chunk()` đã dùng sẵn - có thể lệch ranh giới vài chục mili-giây, cùng loại rủi ro đã ghi ở mục Limitations của README, không ảnh hưởng đáng kể chất lượng transcribe.

Đã test thật bằng ffmpeg thật (9.3): dựng video giả (`lavfi`) 6 giây, hạ ngưỡng chunking xuống 1 giây để ép chạy đúng nhánh chia song song + ghép nối, xác nhận audio ghép lại đúng tổng thời lượng gốc.

### 9.3 Bộ test mới: `process/tests/` (unit + integration, đã chạy thật — 72/72 pass)

Theo yêu cầu, test được tổ chức theo nhóm dưới `process/tests/`:

- **`unit/`** (64 test, không cần ffmpeg/GPU) — `test_config.py` (parse `.env`, validate chunk settings), `test_batch_state.py` (lock + recovery + retry, mục 9.1), `test_speaker_id.py` (cosine similarity, match theo threshold, fallback "Speaker N", đọc/ghi hồ sơ), `test_transform_chunking.py` (`plan_chunks()` thuần tuý - không silence, có silence trong/ngoài cửa sổ cho phép), `test_load.py` (định dạng file markdown cuối cùng - đúng phần bạn cần ở đầu ra).
- **`integration/`** (8 test, cần ffmpeg thật, tự bỏ qua nếu máy không có ffmpeg) — `test_ffmpeg_pipeline.py`: `probe_media_info` (đọc đúng duration, báo lỗi đúng khi không có audio/file không tồn tại), `extract_audio` cả 2 nhánh (single-pass và chunked/song song, mục 9.2), `cut_chunk`, `detect_silences` (dựng audio giả có khoảng lặng chèn sẵn, xác nhận phát hiện đúng vị trí).

Chạy: `cd process && python -m unittest discover -s tests -p "test_*.py" -v` — xem `process/tests/README.md` để chạy từng nhóm/từng file riêng. Không cần cài thêm gì ngoài `numpy` (đã có sẵn trong `requirements.txt`).

**Đã tự kiểm tra được ở vòng này** (máy review không có GPU/torch/whisper/pyannote, xem lại ở dưới): `py_compile` toàn bộ file thay đổi, import thật từng module (trừ `main.py`/`batch.py` - xem ghi chú tkinter bên dưới), toàn bộ 72 test trong `process/tests/` chạy pass, rà tay `batch.py` khớp đúng chữ ký `run_pipeline(video_path, log) -> output_path` của `main.py`.

**Chưa/không thể tự kiểm tra được** — cần bạn chạy thật:
- Toàn bộ phần model ML thật (Whisper, pyannote diarization + embedding) - máy review không cài `torch`/`whisper`/`pyannote.audio` và không có GPU passthrough, như đã nêu ở vòng 1.
- `main.py` và `batch.py` không tự import được trên máy review vì thiếu `tkinter` (module GUI, sandbox Linux tối giản không có sẵn) - đây là hạn chế môi trường review, không phải lỗi mới; máy Windows thật của bạn có `tkinter` sẵn trong bản cài Python chuẩn nên sẽ không gặp lỗi này. Đã bù lại bằng cách kiểm tra tay: `batch.py` gọi đúng `run_pipeline()` với đúng tham số, và mọi module nó phụ thuộc (`pipeline.batch_state`, `pipeline.utils`, `config`) đều import + test thật thành công.
- `speaker_id.py` cụ thể vẫn cần bạn có `HUGGINGFACE_TOKEN` đã accept điều khoản 2 model rồi chạy thử enroll + 1 video thật để xác nhận độ chính xác matching và tune `SPEAKER_MATCH_THRESHOLD` (như đã nêu ở vòng 1).
- `batch.py` chạy thật với nhiều video + Task Scheduler thật trên máy bạn - logic đã unit test kỹ (9.1), nhưng chưa có lần chạy end-to-end thật với video thật.

**Vòng cập nhật thứ 3 (cùng ngày, theo yêu cầu tiếp theo của bạn)** — bạn gửi một transcript output thật từ lần chạy trước để mình đánh giá chất lượng, hỏi thêm về các case chưa test (đa ngôn ngữ trong 1 video, giọng nói chồng lấn, từ chuyên ngành phải nói bằng tiếng Anh xen giữa câu tiếng Việt), rồi **từ chối đề xuất ban đầu của mình** — một từ điển sửa lỗi thủ công — với lý do đúng: không thể biết trước người nói sẽ phát âm sai từ nào để liệt kê hết vào từ điển. Theo hướng bạn chỉ định ("cứ lưu data raw... xong cung cấp ngữ cảnh cho mô hình LLM để kiểm soát đầu ra"), mình đổi sang giải pháp sửa lỗi bằng LLM đọc ngữ cảnh (dùng chính `ANTHROPIC_API_KEY` đã có sẵn), sau đó bạn yêu cầu triển khai toàn bộ test case đã đề xuất ("ok thêm vào test case hết đi"). Chi tiết ở 9.4-9.5 dưới đây.

### 9.4 Module mới: `pipeline/enrich/correct.py` (sửa lỗi transcript bằng LLM, đọc ngữ cảnh)

Vấn đề: Whisper thỉnh thoảng phiên âm sai tên riêng, thuật ngữ tiếng Anh xen giữa câu tiếng Việt, hoặc từ chuyên ngành nó không biết - và `DOMAIN_VOCABULARY` (hint tĩnh đưa trước cho Whisper) không giúp được gì cho từ mà chính bạn cũng không liệt kê trước được ("Tôi ko nhớ hết được những từ sẽ xuất hiện" - đúng như bạn nói).

`correct_transcript_errors(segments, video_path, domain_vocabulary, api_key, model, log)` chạy ngay sau `transform_to_segments()`, trước diarization/topic segmentation. Gửi Claude toàn bộ transcript trong **1 lần gọi duy nhất** (mỗi dòng đánh số `[i] text`), kèm tên file video + `DOMAIN_VOCABULARY` (nếu có) làm ngữ cảnh, và yêu cầu trả về **chỉ những dòng nó tin là bị sai** dưới dạng JSON `[{"index":, "text":}]` - prompt cấm rõ ràng việc diễn giải lại/tóm tắt/dịch/đổi nghĩa câu nói, chỉ được sửa từ sai. Nếu không có gì sai, Claude trả về `[]`. Giống mọi bước tuỳ chọn khác trong codebase (diarization, topic segmentation), lỗi bất kỳ (key sai, mất mạng, JSON hỏng) đều được bắt lại, log cảnh báo, và trả nguyên transcript gốc - không bao giờ làm hỏng pipeline chính. Mọi `index` Claude trả về đều được kiểm tra nằm trong khoảng hợp lệ trước khi áp dụng; chỉ dòng nào text sau sửa thực sự khác dòng gốc (so sánh sau khi strip) mới tính là "đã sửa" trong log cuối.

**Một điểm cần nói rõ với bạn: thiết kế cuối cùng khác với những gì mình mô tả ban đầu.** Lúc đề xuất, mình có nói response sẽ khớp 1-1 với số lượng segment (dùng số lượng không khớp làm tín hiệu phát hiện lỗi). Trong lúc code thật, mình đổi sang kiểu "chỉ trả về dòng cần sửa" - đúng pattern `segment_topics()` đã dùng sẵn trong `pipeline/enrich/enrich.py` - và validate bằng cách kiểm tra từng `index` có hợp lệ, thay vì so số lượng. Lý do: rẻ hơn nhiều cho trường hợp phổ biến (transcript đúng gần hết, ví dụ file 96 dòng bạn gửi có lẽ chỉ 0-2 lỗi thật) - Claude chỉ cần trả vài dòng JSON thay vì lặp lại cả 96 dòng; và an toàn hơn (không có cách nào áp 1 response bị hỏng/thiếu vào sai vị trí). Đây là một cải tiến hợp lý theo mình, nhưng là **hợp đồng khác** với điều đã nói trước đó, nên nêu lại ở đây - báo mình nếu bạn muốn quay về kiểu match 1-1 đầy đủ thay vì sparse response này.

Wiring vào `main.py`: gọi ngay sau `transform_to_segments()`, dùng chung `ANTHROPIC_API_KEY`/`ANTHROPIC_MODEL` với topic segmentation - cả 2 cùng bật/tắt theo 1 key, mỗi bên tự log 1 dòng riêng khi bị bỏ qua.

### 9.5 Sửa lỗi để `main.py`/`batch.py` tự test được + 46 test mới (72 → 118 test, pass 118/118)

`main.py` trước đây `import tkinter` (và `from pipeline.gui import App`, module này cũng tự `import tkinter`) ở top-level file - nên trên máy review (không có `tkinter`) không cách nào import được `main.py`/`batch.py`, nói gì đến unit test `run_pipeline()`. Đây là gap đã nêu rõ ở mục 9.3 ("main.py và batch.py không tự import được trên máy review vì thiếu tkinter"). Đã chuyển cả 2 import đó vào bên trong `main()` (nơi duy nhất chúng được dùng) - không đổi hành vi khi chạy `python main.py` thật, đã xác nhận lại bằng `py_compile` + `import main; import batch` thật trên máy review. **Gap ở 9.3 coi như đã đóng.**

Việc này mở khoá `tests/unit/test_run_pipeline.py` (9 test) - mock toàn bộ hàm `run_pipeline()` gọi tới bằng `unittest.mock.patch("main.<tên hàm>")`, xác nhận: đúng thứ tự gọi khi cả 2 bước tuỳ chọn đều bật; `HUGGINGFACE_TOKEN`/`ANTHROPIC_API_KEY` bật/tắt đúng đúng nhóm hàm; thư mục tạm bị xoá khi chạy xong bình thường; thư mục tạm được giữ lại khi `KEEP_TEMP_FILES=true`; và quan trọng nhất - thư mục tạm **vẫn bị xoá** và exception **vẫn được raise tiếp** khi 1 bước giữa chừng lỗi (đây chính là hợp đồng mà `mark_failed()`/`mark_done()` của `batch.py` ở mục 9.1 phụ thuộc vào).

Các test khác cũng thêm trong vòng này:
- `test_diarize.py` (8 test) - phần logic thuần tuý `assign_speakers()` chưa từng có test: 1 segment nằm hẳn trong 1 turn, segment chồng lấn 2 turn (chọn overlap lớn hơn), không overlap giữ `None`, 2 khoảng chạm biên tính overlap = 0, hoà điểm giữ turn xuất hiện trước, nhiều segment độc lập, xác nhận sửa in-place.
- `test_correct_transcript.py` (13 test) - module mới ở 9.4, giả lập package `anthropic` (chưa cài trong môi trường này) bằng `unittest.mock.patch.dict(sys.modules, ...)` - không chạm mạng thật: chỉ sửa đúng index Claude trả về, index âm/vượt giới hạn tự rơi về bản gốc, JSON hỏng tự rơi về bản gốc, lỗi gọi API không bao giờ raise lên trên, list rỗng thì không gọi API, response `"[]"` hợp lệ giữ nguyên transcript, text sửa giống hệt bản gốc không tính là "đã đổi", response bọc trong \`\`\`json vẫn parse được, prompt có/không có tên file và domain vocabulary đúng như cấu hình.
- `test_batch_helpers.py` (16 test) - các hàm thuần tuý của `batch.py` chưa từng có test: `find_video_files` (quét đệ quy + lọc đúng đuôi file), `is_file_stable` (phát hiện file đang copy dở qua 2 lần đo size), `clean_orphaned_temp_dirs` (dọn thư mục tạm `video2text_*`/`enroll_*` sót lại từ lần chạy bị crash - cùng tinh thần khôi phục sau crash ở mục 9.1, nhưng cho file tạm thay vì `batch_state.json`), `make_file_logger` (ghi log có timestamp ra nhiều file đích cùng lúc).

Tổng: 72 test cũ + 46 test mới = **118 test**, chạy `python tests/run_all.py -v` thật, **118/118 pass**.

### 9.6 `output/` tách thành `raw/` và `processed/` - audio thô không còn là file tạm bị xoá

Theo yêu cầu của bạn sau khi test thật `try_extract.py` trên video dài 2 tiếng (`DA_Buổi_1.mp4`, 7162.3s - xác nhận đúng nhánh trích xuất song song theo chunk hoạt động chính xác, duration khớp tuyệt đối): `output/raw/` lưu audio thô EXTRACT tách ra (`<tên_video>.wav`), `output/processed/` lưu transcript Markdown cuối cùng - đúng tinh thần "raw vs processed" của ETL mà project này đã theo từ đầu.

Trước đây `audio.wav` chỉ là file tạm trong `tempfile.mkdtemp()`, bị xoá ngay sau mỗi lần chạy (trừ khi `KEEP_TEMP_FILES=true`) - nghĩa là muốn chạy lại TRANSFORM/ENRICH trên cùng 1 video (đổi `MODEL_SIZE`, thử lại diarization...) phải giải mã lại từ đầu video gốc. Giờ `RAW_AUDIO_DIR`/`PROCESSED_DIR` (`config.py`) thay cho `OUTPUT_DIR` cũ; `main.py` lưu audio vào `output/raw/<tên_video>.wav` - **ghi đè theo tên video, không theo ngày** (khác với transcript Markdown vẫn ghi đè theo ngày như cũ) - vì audio trích ra từ cùng 1 video luôn giống hệt nhau bất kể chạy lúc nào, đánh dấu ngày ở đây chỉ tổ chiếm ổ cứng vô ích với file có thể vài trăm MB (video test 2 tiếng ra file 218MB). File `chunk_*.wav` của Whisper (TRANSFORM) vẫn là file tạm bị xoá như cũ, không đổi.

`extract_audio()` (`pipeline/extract/extract.py`) được thêm `os.makedirs(os.path.dirname(audio_path), exist_ok=True)` ở đầu hàm - tự tạo thư mục đích nếu chưa có (phòng trường hợp clone project mới chưa có sẵn `output/raw/`), không phụ thuộc caller phải tự tạo trước.

Đã test lại: `try_extract.py` chạy thật lần 2 trên đúng video 2 tiếng đó, xác nhận file ra đúng `output/raw/DA_Buổi_1.wav` (229,192,438 byte, khớp lại với duration gốc, diff 0.0s). Toàn bộ 118 test cũ vẫn pass nguyên (không có test nào phụ thuộc vào `OUTPUT_DIR` cũ theo tên - `test_load.py` luôn truyền `output_dir` riêng qua tham số, `test_run_pipeline.py` mock toàn bộ `extract_audio`/`write_markdown` nên không chạm filesystem thật).

### 9.7 2 lỗi thật tìm thấy khi chạy trên máy bạn — sửa + test lại (137 test)

Bạn chạy `try_pipeline.py` thật trên đoạn clip 3 phút (`.env` đã điền `HUGGINGFACE_TOKEN` + `ANTHROPIC_API_KEY`) và phát hiện 2 lỗi thật mà toàn bộ 118 test trước đó không bắt được:

**Lỗi 1 - `TypeError: 'ThinkingBlock' object has no attribute 'text'`** (transcript correction) — `correct.py`/`enrich.py` đều giả định `response.content[0]` luôn là block text, nhưng khi Claude dùng "extended thinking", block đầu tiên là `ThinkingBlock` (không có `.text`). `correct.py` tự bắt được lỗi này (có try/except, log "skipped") nên không crash - nhưng đây vẫn là 1 bug thật cần sửa vì mọi lần gọi Claude sau này (khi bật thinking) sẽ luôn bị skip. `enrich.py` có **y hệt lỗi** nhưng chưa kịp lộ ra do pipeline đã crash trước đó ở lỗi 2.

Sửa: thêm `extract_text_from_anthropic_response()` vào `pipeline/utils.py` - duyệt qua toàn bộ `response.content`, chỉ lấy các block có `type == "text"`, bỏ qua block khác (thinking, v.v.) thay vì giả định vị trí `[0]`. Cả `correct.py` và `enrich.py` dùng chung hàm này.

**Lỗi 2 - `TypeError: Pipeline.from_pretrained() got an unexpected keyword argument 'use_auth_token'`** (diarization) — pyannote.audio bản mới trên máy bạn đã đổi tên tham số `use_auth_token` thành `token` (theo huggingface_hub). Nghiêm trọng hơn: **`main.py` không hề bọc try/except quanh khối diarization** như đã làm với correction/topic segmentation - nên lỗi này làm sập toàn bộ pipeline, **mất luôn 5 phút Whisper vừa transcribe xong** (đúng như bạn gặp thật). Đây là chỗ code không nhất quán với chính triết lý "never crash pipeline" đã áp dụng cho các bước tuỳ chọn khác trong project.

Sửa 2 phần:
- `diarize.py`: `Pipeline.from_pretrained()` thử `token=` trước (bản pyannote.audio hiện tại), nếu `TypeError` thì tự fallback sang `use_auth_token=` (bản cũ hơn) - không cần biết trước máy nào cài bản nào.
- `main.py`: bọc try/except quanh toàn bộ khối diarization + speaker recognition - lỗi ở đây giờ chỉ log `"Speaker diarization skipped (failed: ...)"` rồi tiếp tục sang topic segmentation + ghi markdown, không còn mất công sức Whisper đã làm.

**Test mới (19 test, 118 → 137)**:
- `test_utils.py` (mới, 5 test) - `extract_text_from_anthropic_response()`: 1 block text, thinking-rồi-text (đúng bug thật), nhiều block text nối lại, không có block text nào, block thiếu hẳn attribute `.type`.
- `test_correct_transcript.py` - cập nhật fake response cho đúng có `type="text"` (bản test cũ vô tình không bắt được bug này vì `MagicMock` tự sinh `.type` giả, không phải chuỗi `"text"` thật), thêm 1 test regression đúng kịch bản ThinkingBlock đứng trước.
- `test_enrich.py` (**mới hoàn toàn**, 8 test) - `segment_topics()` trước đây chưa từng có test riêng dù đã có từ những vòng đầu của project. Cùng mức test rigor như `test_correct_transcript.py`, gồm cả regression test ThinkingBlock.
- `test_diarize.py` - thêm class `TestDiarizeAudioParamCompat` (4 test): `token=` thành công ngay, fallback `use_auth_token=` khi `token=` bị `TypeError`, xác nhận `TypeError` từ chính lúc chạy diarization (không phải từ `from_pretrained()`) vẫn propagate bình thường chứ không bị nhầm là cần fallback, nhiều turn/nhiều speaker đi qua đúng. Dùng kỹ thuật `sys.modules` injection cho module có dấu chấm (`pyannote.audio`) - lần đầu áp dụng kỹ thuật này cho 1 package con thay vì package gốc như `anthropic`.
- `test_run_pipeline.py` - thêm `test_diarization_failure_is_caught_and_pipeline_continues`: diarization lỗi → `assign_speakers`/`load_profiles` không được gọi, nhưng `write_markdown` vẫn chạy và `run_pipeline()` vẫn trả về bình thường, không raise.

Đã chạy lại toàn bộ `tests/run_all.py -v`: **137/137 pass**. `try_pipeline.py` sẽ được chạy lại thật trên máy bạn để xác nhận cả 2 lỗi đã hết (kết quả sẽ cập nhật sau khi bạn chạy).

### 9.8 Log token usage thật cho mỗi lần gọi Claude (137 → 139 test)

Bạn hỏi "lượng token tốn là bao nhiêu" - code trước đó không hề log usage thật, chỉ có thể ước tính. Thêm `log_anthropic_usage(response, log)` vào `pipeline/utils.py` - đọc `response.usage.input_tokens`/`.output_tokens` (luôn có mặt trên response thật từ SDK `anthropic`), log 1 dòng `"Claude usage: N input + M output tokens"`. Gọi ngay sau mỗi `client.messages.create(...)` ở cả `correct.py` và `enrich.py`, trước khi parse response - nên vẫn log được ngay cả khi parse thất bại sau đó (đúng như log thật sau này cho thấy ở 9.10, dòng usage xuất hiện trước dòng "skipped" khi correction lỗi).

Test mới trong `test_utils.py`: log đúng khi có `usage`, không log/không raise khi response không có attribute `usage` (phòng trường hợp SDK hoặc bản mock cũ chưa có field này). 137 → 139 test, `run_all.py -v` pass.

### 9.9 Diarization (pyannote) cũng tự chuyển sang GPU nếu có - trước đây luôn chạy CPU (139 → 141 test)

Bạn hỏi có cách nào chạy nhanh hơn không, gửi ảnh xác nhận máy có GPU rời thật (NVIDIA RTX 3060 Laptop, 6GB VRAM riêng - "GPU 1" trong Task Manager, khác "GPU 0" là Intel UHD tích hợp không dùng được CUDA). Đọc lại code phát hiện: `transform.py` (Whisper) đã tự nhận `device = "cuda" if torch.cuda.is_available() else "cpu"` từ trước, nhưng `diarize.py` (pyannote) thì **không hề** - `Pipeline.from_pretrained()` luôn load lên CPU, dòng log cũ còn ghi cứng "this can take a while on CPU" bất kể máy có GPU hay không. Đây là một khoảng trống thật, không phải lỗi mới phát sinh.

Sửa: thêm cùng logic `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` rồi `pipeline = pipeline.to(device)` ngay sau khi load pipeline, trước khi chạy diarization - không đổi hành vi nếu máy không có CUDA torch (tự rơi về CPU y như cũ). Test mới trong `test_diarize.py`: thêm helper giả `torch` (`_fake_torch_module`) vì môi trường review không cài `torch` thật; `instance.to.return_value = instance` để mock phản ánh đúng hành vi thật của `.to(device)` (mutate in-place, trả về chính nó) - nếu không sẽ làm 4 test cũ của `TestDiarizeAudioParamCompat` fail vì `pipeline(audio_path)` sau đó gọi nhầm vào 1 mock con chưa cấu hình; thêm 2 test mới xác nhận đúng `.to()` được gọi với `cpu`/`cuda` tương ứng. 139 → 141 test, pass.

**Lưu ý quan trọng: việc này không đánh đổi độ chính xác** - vẫn đúng model, đúng phép tính, chỉ chạy trên phần cứng khác. Bạn cần tự cài lại `torch` bản CUDA trên máy Windows thật (`pip install torch --index-url https://download.pytorch.org/whl/cu126`) - không làm được từ môi trường cầu nối review.

### 9.10 Chạy thật full pipeline trên video gốc 2 tiếng — lỗi thật thứ 3: correction thất bại hoàn toàn trên video dài, sửa bằng cách chia batch (141 → 148 test)

Bạn chạy `try_pipeline.py` thật trên `DA_Buổi_1.mp4` (7162s, ~2 tiếng) sau khi các thay đổi ở 9.8-9.9. Kết quả log thật:

- **TRANSFORM (Whisper)**: chạy CPU (`Using device: cpu` - torch CUDA của 9.9 chưa được cài vào lúc chạy lần này), 24 chunk, tổng **10816.9s (~3 tiếng)**.
- **CORRECTION: thất bại hoàn toàn.** `Claude usage: 43786 input + 8000 output tokens` rồi ngay sau đó `Transcript correction skipped (Claude call failed: Expecting value: line 1 column 1 (char 0))`. Output đúng bằng **8000** - đúng giới hạn `max_tokens` vừa tăng gấp đôi (4000→8000) ở phiên trước - nghĩa là dù đã tăng cap, vẫn bị cắt. Lỗi `"Expecting value: line 1 column 1 (char 0)"` là dấu hiệu đặc trưng của `json.loads("")` - chuỗi rỗng. Kết luận: với transcript ~1500 dòng dồn vào 1 lần gọi, model dùng **toàn bộ** ngân sách output cho phần "suy nghĩ" (extended thinking) nội bộ, không còn chỗ để viết ra JSON trả lời - `response.content` chỉ có `ThinkingBlock`, không có block `text` nào, nên `extract_text_from_anthropic_response()` trả về `""`. Đây chính là lỗi bạn nghi ngờ khi hỏi "bản này chưa chuẩn hoá API Anthropic phải k" về file transcript ngày 23-09-26 - **đúng, đã xác nhận bằng log thật**: correction chưa bao giờ chạy thành công cho video đó.
- **DIARIZATION: thất bại** (403, không liên quan đến bug trên) - `Access to model pyannote/speaker-diarization-community-1 is restricted`. Đây là do tài khoản HuggingFace đứng sau `HUGGINGFACE_TOKEN` **chưa accept điều khoản** của model gated này trên trang https://hf.co/pyannote/speaker-diarization-community-1 - cần bạn tự đăng nhập đúng tài khoản đó và bấm chấp nhận, không phải lỗi code.
- **TOPIC SEGMENTATION: thành công.** `Claude usage: 43635 input + 2403 output tokens` → 30 topic - khớp đúng 30 tiêu đề `##` mà bạn thấy trong file đã upload. Cap `enrich.py` vừa tăng (2000→4000 ở phiên trước) vừa đủ dư - nếu vẫn giữ 2000 cũ thì bước này (2403 output) **cũng đã fail theo đúng cách tương tự**.

**Bài học rút ra: chỉ tăng `max_tokens` không phải là fix đúng** - vì "suy nghĩ" nội bộ của model tăng theo độ phức tạp/độ dài input, tăng cap mãi vừa tốn tiền vừa không có điểm dừng an toàn cho video càng lúc càng dài hơn. Cách sửa đúng: **chia transcript thành nhiều batch nhỏ** cho bước correction, giống hệt cách `transform.py` đã chia chunk cho Whisper từ đầu - mỗi lần gọi Claude chỉ xử lý 1 phần nhỏ transcript, giữ input/thinking/output luôn nhỏ và ổn định bất kể video dài bao nhiêu.

**Thay đổi trong `correct_transcript_errors()`** (`pipeline/enrich/correct.py`): thêm tham số `batch_size` (mặc định 150, cấu hình qua `CORRECTION_BATCH_SIZE` trong `.env`/`config.py`). Transcript được chia thành các batch `batch_size` dòng, mỗi batch 1 lần gọi Claude riêng, đánh số dòng theo **chỉ số toàn cục** (`_build_numbered_transcript(..., start_index=batch_start)`) để index Claude trả về map thẳng vào segment gốc, không cần remap. Một batch lỗi (network, JSON hỏng, response rỗng như lỗi thật vừa gặp, index ngoài phạm vi) chỉ log cảnh báo và bỏ qua batch đó (`continue`) - **không còn huỷ toàn bộ correction của cả video** như thiết kế cũ (1 lỗi = huỷ hết). Việc kiểm tra index hợp lệ (`0 <= idx < len(segments)`) vẫn giữ nguyên logic phòng thủ cũ, chỉ thu hẹp phạm vi ảnh hưởng xuống 1 batch thay vì cả response. `main.py` truyền `batch_size=CONFIG["correction_batch_size"]` vào lời gọi.

**Test mới (7 test) trong `test_correct_transcript.py`, class `TestCorrectTranscriptErrorsBatching`**: transcript nhỏ hơn batch_size → đúng 1 lần gọi (giữ nguyên hành vi cũ, 14 test cũ đều pass không cần sửa gì); transcript lớn hơn → đúng số lần gọi tương ứng; batch thứ 2 đánh số dòng bắt đầu từ chỉ số toàn cục (không phải `[0]`); correction từ mọi batch đều được áp dụng đúng; 1 batch lỗi mạng không làm mất correction của batch khác; 1 batch trả index ngoài phạm vi không ảnh hưởng batch khác; và quan trọng nhất - **regression test đúng y hệt lỗi thật vừa gặp** (response không có block `text` nào → `extract_text_from_anthropic_response()` trả `""` → batch đó bị bỏ qua, nhưng batch khác vẫn áp dụng bình thường).

Đã chạy lại toàn bộ `tests/run_all.py -v`: **148/148 pass**. Batch size mặc định 150 dòng: với video 2 tiếng thật (~1500 dòng, 43786 token input cho cả transcript / ~1500 dòng ≈ 29 token/dòng), mỗi batch ~150 dòng chỉ khoảng ~4300 token input - dư sức nằm trong ngân sách 8000 token output kể cả phần thinking. Chưa chạy lại thật trên video gốc để xác nhận correction chạy hết cả 2 tiếng không lỗi batch nào - cần bạn chạy `try_pipeline.py` lại lần nữa.

### 9.11 Diarization lỗi thật thứ 4: `torchcodec` không load được DLL trên Windows - sửa bằng cách bỏ qua torchcodec hoàn toàn (149 test)

Sau khi bạn accept điều khoản HuggingFace (9.10 đóng lại), diarization tải model thành công (`config.yaml`, `segmentation/pytorch_model.bin`, `plda/*.npz`, `embedding/pytorch_model.bin`) nhưng gặp lỗi mới khi thực sự chạy:

```
Speaker diarization skipped (failed: Could not load libtorchcodec. Likely causes:
  1. FFmpeg is not properly installed...
  2. The PyTorch version (2.14.0+cpu) is not compatible with this version of TorchCodec...
```

Nguyên nhân: gọi `pipeline(audio_path)` (truyền đường dẫn file) khiến pyannote.audio tự giải mã audio nội bộ qua `torchcodec` - một thư viện Meta mới, **khác hoàn toàn** với `ffmpeg.exe` project này đã dùng cho EXTRACT. `torchcodec` cần bộ DLL FFmpeg dạng "shared" (không phải bản `ffmpeg.exe` đơn lẻ thường tải trên Windows) để `ctypes` load động lúc chạy - máy bạn không có bộ DLL đó nên load thất bại ở mọi phiên bản FFmpeg nó thử (4 đến 9).

Đã tra cứu kỹ trước khi sửa (tài liệu chính thức của model + các issue thật trên GitHub `pyannote-audio`/`torchcodec`) thay vì đoán: cách sửa đúng không phải là cài thêm FFmpeg "full-shared" build + chỉnh PATH (phức tạp, dễ vỡ, phải cài lại mỗi khi đổi máy) mà là **tránh dùng torchcodec hoàn toàn** - model card chính thức của `pyannote/speaker-diarization-community-1` trên HuggingFace xác nhận pipeline nhận cả 2 kiểu input: đường dẫn file (kích hoạt torchcodec) hoặc dict `{"waveform": tensor, "sample_rate": int}` đã load sẵn bằng `torchaudio.load()` (không đụng torchcodec).

Sửa trong `diarize_audio()` (`pipeline/enrich/diarize.py`): thêm `import torchaudio`, thay `pipeline(audio_path)` bằng:
```python
waveform, sample_rate = torchaudio.load(audio_path)
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
```
`torchaudio` đã có sẵn (dependency bắc cầu của `pyannote.audio`), không cần cài thêm gì.

Test: thêm `_fake_torchaudio_module()` vào `test_diarize.py` (cùng kỹ thuật `sys.modules` injection như `torch`/`pyannote.audio`), gắn vào cả 6 test của `TestDiarizeAudioParamCompat` cần nó; thêm 1 test regression mới xác nhận `torchaudio.load()` được gọi đúng với `audio_path`, và pipeline được gọi với **dict** `{"waveform":, "sample_rate":}` chứ không phải chuỗi đường dẫn thô - khoá đúng hành vi vừa sửa. 149/149 pass.

Chưa chạy lại thật trên máy bạn để xác nhận lỗi `libtorchcodec` đã hết - cần bạn chạy lại `try_pipeline.py` (clip ngắn trước) sau khi pull thay đổi này.

### 9.12 Sửa lại 9.11 - `torchaudio.load()` KHÔNG né được torchcodec như đã nghĩ, đổi sang `soundfile` (149 test)

Bạn chạy lại clip ngắn sau bản sửa 9.11, **vẫn dính đúng lỗi `libtorchcodec` y hệt** - chỉ khác ở chỗ thông báo giờ ghi `"Failed to create AudioDecoder for ...DA_Buổi_1_short.wav"`. Đây là bằng chứng fix ở 9.11 sai: `torchaudio.load()` **không né được torchcodec** trên phiên bản `torchaudio` đi kèm `torch 2.14.0+cpu` của bạn - các bản `torchaudio` gần đây (≈2.9 trở lên) đã chuyển sang **tự dùng torchcodec làm backend nội bộ**, nên gọi `torchaudio.load()` vẫn kích hoạt đúng lỗi DLL cũ, chỉ đi vòng thêm 1 lớp. Đã kiểm chứng lại qua nhiều nguồn (issue thật trên GitHub của `torchaudio`/`torchcodec`/dự án khác gặp đúng lỗi này) trước khi sửa lần 2, không đoán mò tiếp.

**Sửa đúng lần này:** dùng `soundfile` (thư viện `libsndfile`, hoàn toàn tách biệt khỏi FFmpeg/torchcodec, có sẵn DLL đóng gói trong wheel Windows - không cần cài thêm gì trên máy) để tự đọc file `.wav` thành mảng số, rồi mới đưa vào `pipeline({"waveform":, "sample_rate":})`. An toàn để dùng ở đây vì `diarize_audio()` **luôn luôn** chỉ nhận file `.wav` do chính EXTRACT stage của project này tạo ra (16kHz mono 16-bit PCM - xem `pipeline/extract/extract.py`), không phải định dạng bất kỳ cần bộ giải mã đa năng như torchcodec/FFmpeg.

```python
data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
waveform = torch.from_numpy(data.T)  # (samples, channels) -> (channels, samples)
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
```

Thêm `soundfile` vào `requirements.txt` (dependency mới, nhẹ, có wheel sẵn cho Windows). Cập nhật `test_diarize.py`: đổi fake `torchaudio` (9.11) thành fake `soundfile` - dùng numpy array thật (không phải `MagicMock`) vì code gọi `.T` lên kết quả; cập nhật test regression để xác nhận đúng `sf.read(audio_path, dtype="float32", always_2d=True)` được gọi và pipeline nhận đúng dict `{"waveform": <mảng đã transpose>, "sample_rate":}`. 149/149 pass.

**Bài học đáng nói thẳng:** fix ở 9.11 dựa trên tài liệu/case study viết cho phiên bản `torchaudio` cũ hơn, không kiểm tra được hành vi thật trên phiên bản mới bạn đang cài (không thể test được từ môi trường review vì không có `torch`/`torchaudio` thật) - nên đưa ra 1 fix tưởng đúng nhưng không né được vấn đề gốc. Lần này dùng `soundfile` triệt để tách khỏi cả torchaudio lẫn torchcodec nên không còn phụ thuộc vào việc phiên bản nào dùng backend nào nữa. Vẫn cần bạn chạy lại `try_pipeline.py` (clip ngắn) để xác nhận lần này thật sự hết lỗi.

### 9.13 Lỗi thật thứ 5: pyannote.audio 4.x bọc kết quả trong `DiarizeOutput`, không trả `Annotation` trực tiếp nữa (151 test)

Fix `soundfile` ở 9.12 **đã đúng** - lỗi `libtorchcodec` biến mất hoàn toàn, diarization chạy thật (~3 phút tính toán trên CPU cho clip 3 phút, thấy rõ qua thời gian `[370.8s]` → `[548.9s]`). Nhưng ngay sau khi chạy xong lại lỗi tiếp, khác hẳn 4 lỗi trước:

```
Speaker diarization skipped (failed: 'DiarizeOutput' object has no attribute 'itertracks')
```

Tra lại thấy: `pip install -r requirements.txt` trên máy bạn đã kéo về `pyannote.audio 4.0.7` (log `pip install` xác nhận), bản này **đổi kiểu dữ liệu trả về** của pipeline - trước đây gọi `pipeline(...)` trả thẳng 1 `Annotation` (có `.itertracks()`), giờ trả về 1 dataclass `DiarizeOutput` bọc bên ngoài, `Annotation` thật nằm ở thuộc tính `.speaker_diarization` (xác nhận qua tài liệu chính thức của model trên HuggingFace: `for turn, speaker in output.speaker_diarization`).

Sửa trong `diarize_audio()`: bóc `Annotation` ra bằng `getattr(result, "speaker_diarization", result)` - nếu `result` có `.speaker_diarization` (pyannote 4.x) thì dùng nó, không thì coi `result` chính là `Annotation` luôn (pyannote cũ hơn, trước 4.0) - cùng tinh thần với cách xử lý tương thích `token=`/`use_auth_token=` đã làm ở 9.7, không cần biết trước máy nào cài bản nào.

Test: cập nhật `_fake_pipeline_instance()` để nhận thêm tham số `wrap_in_diarize_output` - mô phỏng đúng 2 kiểu trả về (bọc/không bọc); phải dùng `del mock.speaker_diarization` cho nhánh "không bọc" vì `MagicMock` tự sinh thuộc tính bất kỳ khi truy cập, nếu không xóa thì `getattr(..., default)` sẽ không bao giờ rơi vào nhánh fallback thật sự (bug tiềm ẩn trong chính cách viết test, không phải trong code chạy thật - đáng chú ý). Thêm 2 test mới: 1 cho kiểu bọc (`DiarizeOutput`, pyannote 4.x - đúng bug thật vừa gặp), 1 cho kiểu không bọc (Annotation trực tiếp, pyannote cũ) - đảm bảo sửa 1 bên không phá bên kia. 151/151 pass.

Tổng cộng đã tìm + sửa **5 lỗi thật khác nhau** chỉ riêng cho bước diarization qua các lượt chạy thật liên tiếp (403 gated repo → libtorchcodec qua torchaudio → libtorchcodec qua torchcodec trực tiếp → DiarizeOutput) - mỗi lỗi chỉ lộ ra sau khi lỗi trước đó được sửa, đúng kiểu "bóc từng lớp" khi chạy thật trên máy với phiên bản dependency mới hơn nhiều so với lúc code được viết ban đầu. Cần bạn chạy lại `try_pipeline.py` lần nữa để xác nhận diarization ra kết quả đúng (có tên/label speaker trong file markdown).


### 9.14 Lỗi thật thứ 6 (cùng họ torchcodec): bước tính voice embedding cũng dính lỗi y hệt diarization - gộp fix dùng chung (151 → 171 test)

Bạn chạy lại `try_pipeline.py` sau khi fix `DiarizeOutput` ở 9.13 - lần này **diarization chạy đúng hoàn toàn**: `Diarization found 1 speaker(s) across 26 turn(s).` (xác nhận cả fix `soundfile` ở 9.12 lẫn fix `DiarizeOutput` ở 9.13 đều đã đúng). Nhưng ngay bước tiếp theo, "Loading speaker embedding model..." (tính voice embedding để nhận diện người nói qua `speaker_id.py`), lại gặp lại **đúng lỗi `libtorchcodec`** đã sửa ở 9.12 cho diarization - lặp lại nhiều lần (mỗi turn tính embedding bị lỗi 1 lần), kết thúc bằng `Computed voice embeddings for 0/1 speaker(s).`. Pipeline không sập (bước này vốn đã có try/except từng turn từ đầu), nhưng toàn bộ chức năng nhận diện người nói qua tên thực tế không hoạt động - transcript vẫn ra nhưng speaker label chỉ là "Speaker 1" thay vì tên thật đã enroll.

Nguyên nhân giống hệt 9.11/9.12: `pipeline/enrich/speaker_id.py`'s `_embed_turns()` gọi `inference.crop(audio_path, Segment(start, end))` - truyền thẳng **đường dẫn file** (chuỗi) cho mỗi turn cần tính embedding, y hệt cách `diarize_audio()` từng gọi `pipeline(audio_path)` trước khi sửa ở 9.12. `pyannote.audio`'s `Inference.crop()` xử lý input kiểu path giống hệt `Pipeline.__call__()` - đều đi qua `Audio.crop()` nội bộ, và khi input là path thì nó tự giải mã qua `torchcodec` bất kể lời gọi đến từ `Pipeline` hay `Inference`. Đã tra lại mã nguồn thật của `pyannote-audio` trên GitHub (`core/inference.py`, `core/io.py`) để xác nhận trước khi sửa, không đoán: `AudioFile` (kiểu tham số `file` của cả `Pipeline` lẫn `Inference`) chấp nhận `str | Path | IOBase | Mapping` - truyền dict `{"waveform": tensor, "sample_rate": int}` thay vì path khiến `Audio` **bỏ qua hoàn toàn** bước giải mã qua torchcodec, dùng thẳng tensor đã có sẵn - đúng cơ chế đã dùng cho diarization ở 9.12, giờ áp dụng lại cho embedding.

**Sửa bằng cách gộp logic dùng chung, không lặp lại:** thay vì copy nguyên khối `sf.read()`/`torch.from_numpy()` từ `diarize.py` sang `speaker_id.py` lần hai, tách thành 1 hàm chung `load_waveform(audio_path)` trong `pipeline/utils.py` (nơi các hàm chia sẻ giữa các stage khác - `extract_text_from_anthropic_response()`, `log_anthropic_usage()` - đã sẵn có) - trả về đúng dict `{"waveform", "sample_rate"}` mà cả `Pipeline` lẫn `Inference` đều chấp nhận. `diarize_audio()` giờ gọi `pipeline(load_waveform(audio_path))` thay vì tự đọc file; `speaker_id.py` gọi `load_waveform()` **một lần duy nhất cho mỗi video** (trong `compute_speaker_centroids()`, trước vòng lặp qua từng speaker, và trong `enroll_speaker()`) rồi truyền cùng 1 object đã load sẵn xuống `_embed_turns()` cho mọi turn của mọi speaker - vừa sửa đúng lỗi torchcodec, vừa tránh đọc lại file audio từ đĩa nhiều lần một cách lãng phí (trước đây nếu sửa nông theo kiểu "đổi `audio_path` thành gọi `sf.read` ngay trong `_embed_turns()`" thì sẽ đọc lại file mỗi turn - đây là cải tiến đi kèm, không chỉ là fix lỗi).

Cả `compute_speaker_centroids()` lẫn `enroll_speaker()` đều bọc bước `load_waveform()` trong try/except: nếu audio không đọc được (file hỏng/thiếu), `compute_speaker_centroids()` log cảnh báo và trả về `{}` như mọi lỗi khác ở stage này (giữ đúng triết lý "không bao giờ làm hỏng pipeline chính" đã nêu ở đầu file `speaker_id.py`), còn `enroll_speaker()` (CLI riêng, không phải 1 bước trong pipeline chính) để lỗi tự raise lên như cũ vì đây là lệnh chạy tay, cần người dùng biết ngay khi input hỏng.

**Test mới (20 test, 151 → 171)**:
- `test_utils.py`, class `TestLoadWaveform` (2 test) - xác nhận `sf.read()` được gọi đúng tham số (`dtype="float32", always_2d=True`), và kết quả trả về đúng dict `{"waveform", "sample_rate"}` với waveform đã transpose sang `(channels, samples)`.
- `test_diarize.py` - không cần sửa gì: các test cũ của `TestDiarizeAudioParamCompat` (bao gồm test regression "gọi bằng dict, không phải path" ở 9.12) vẫn pass nguyên sau khi đổi sang gọi `load_waveform()` từ `pipeline/utils.py`, vì kỹ thuật giả lập `sys.modules` cho `torch`/`soundfile` không quan tâm module nào thực sự gọi `import torch`/`import soundfile` - chỉ cần đúng module đó được import trong lúc patch còn hiệu lực.
- `test_speaker_id.py` - trước đây file này ghi rõ trong docstring là "không test phần cần model thật" (`_embed_turns`/`compute_speaker_centroids`/`enroll_speaker` chưa từng có test). Giờ áp dụng đúng kỹ thuật `sys.modules` injection đã dùng cho `pyannote.audio` ở `test_diarize.py`, mở rộng thêm cho `pyannote.core.Segment`:
  - `TestEmbedTurns` (7 test) - `inference.crop()` nhận đúng object audio đã preload (không phải path/chuỗi); nhiều turn được tính trung bình đúng; turn ngắn hơn ngưỡng tối thiểu (1.5s) bị bỏ qua khi có turn dài hơn; nếu mọi turn đều ngắn thì vẫn dùng hết (không bỏ hoàn toàn); giới hạn đúng số turn tối đa mỗi speaker (5, ưu tiên turn dài nhất); 1 turn lỗi không làm mất kết quả turn khác; mọi turn đều lỗi thì trả `None`.
  - `TestComputeSpeakerCentroids` (6 test) - mỗi speaker ra đúng 1 centroid; **regression test quan trọng nhất**: file audio chỉ đọc từ đĩa đúng 1 lần dù có nhiều speaker (khoá đúng cải tiến "load 1 lần" vừa nêu trên); turn cùng speaker label được gộp đúng nhóm; model tải lỗi hoặc audio đọc lỗi đều rơi về `{}` chứ không raise.
  - `TestEnrollSpeaker` (5 test) - `inference.crop()` cũng nhận đúng object đã preload (cùng bug, khác điểm gọi - trước giờ enroll_speaker.py's CLI chưa từng được test); enroll người mới tạo đúng profile; enroll lại người cũ gộp đúng theo trung bình có trọng số (`sample_count`); `update_existing=False` thêm profile mới thay vì gộp; không tính được embedding nào thì raise đúng `RuntimeError`.

Đã chạy lại toàn bộ `tests/run_all.py -v`: **171/171 pass**. Tổng cộng đã tìm + sửa **6 lỗi thật khác nhau** cùng họ dependency (pyannote.audio/torch/torchcodec) qua các lượt chạy thật liên tiếp trên máy Windows của bạn - lỗi cuối này chỉ lộ ra sau khi 5 lỗi trước đó (403 gated repo → libtorchcodec qua torchaudio → libtorchcodec qua torchcodec trực tiếp → DiarizeOutput → giờ là embedding) đều đã được sửa, vì bước tính embedding chỉ chạy được sau khi diarization tự nó chạy xong. Cần bạn chạy lại `try_pipeline.py` (clip ngắn trước) để xác nhận `Computed voice embeddings for N/N speaker(s)` (khác `0/N` như log lần trước) - nếu bạn đã enroll sẵn người nói nào trong `speaker_profiles.json`, đây cũng là lúc xác nhận tên thật xuất hiện đúng trong file markdown thay vì "Speaker 1".


### 9.15 Xác nhận fix 9.14 đúng qua log thật + 1 lỗi nhỏ thêm phát hiện được: token HuggingFace không thực sự được dùng khi tải model embedding (171 → 174 test)

Bạn chạy lại `try_pipeline.py` trên clip ngắn sau bản sửa 9.14 - **toàn bộ pipeline chạy thành công từ đầu đến cuối, không còn lỗi nào**:

```
[  360.0s] Corrected 8 likely transcription error(s).
[  583.3s] Diarization found 1 speaker(s) across 26 turn(s).
[  585.9s] Computed voice embeddings for 1/1 speaker(s).
[  589.1s] Found 3 topic(s).
[  589.1s] Done! Result: ...transcript.md
```

`Computed voice embeddings for 1/1 speaker(s)` (thay vì `0/1` như log trước 9.14) xác nhận đúng cả 2 phần của fix 9.14: lỗi `libtorchcodec` đã hết hoàn toàn, và việc gộp logic đọc audio dùng chung qua `pipeline.utils.load_waveform()` hoạt động đúng ở cả 2 chỗ gọi (diarization lẫn embedding).

Trong log này có 1 dòng cảnh báo (không phải lỗi) đáng chú ý: `Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.` xuất hiện ngay trước lúc tải model embedding. Tra lại thấy đây là **cùng họ lỗi đã sửa ở 9.7** (`huggingface_hub` đổi tên tham số `use_auth_token` thành `token`) nhưng ở 1 điểm gọi khác chưa được sửa: `_load_embedding_model()` (`speaker_id.py`) vẫn gọi `Model.from_pretrained(_EMBEDDING_MODEL_NAME, use_auth_token=hf_token)` - khác với `Pipeline.from_pretrained()` trong `diarize.py` đã có fallback `token=`/`use_auth_token=` từ 9.7.

Đã tra lại mã nguồn thật của `pyannote-audio` (`core/model.py` trên GitHub) trước khi sửa: `Model.from_pretrained()` bản hiện tại khai báo rõ tham số `token=` trong signature, nhưng **không** raise lỗi khi nhận `use_auth_token=` như `Pipeline.from_pretrained()` làm - nó âm thầm gom vào `**kwargs` rồi chuyển xuống hàm load checkpoint nội bộ (PyTorch Lightning), nghĩa là token **không hề được dùng để xác thực với HuggingFace Hub**. Vì model `pyannote/wespeaker-voxceleb-resnet34-LM` không bị gate nên tải vẫn thành công (không sập), chỉ là chạy ở chế độ ẩn danh (giới hạn tốc độ thấp hơn, không có lợi ích gì từ `HUGGINGFACE_TOKEN` đã cấu hình) - đúng như cảnh báo nói.

Sửa: áp dụng đúng pattern try/except đã dùng cho `Pipeline.from_pretrained()` ở 9.7 - thử `token=hf_token` trước (đúng tên hiện tại), rơi về `use_auth_token=hf_token` nếu `TypeError` (bản `pyannote.audio` cũ hơn không có tham số `token=`). Không cần biết trước máy nào cài bản nào, cùng tinh thần nhất quán với `diarize.py`.

Test mới (3 test, class `TestLoadEmbeddingModelParamCompat` trong `test_speaker_id.py`): dùng `token=` thành công ngay không cần fallback; fallback đúng sang `use_auth_token=` khi `token=` bị `TypeError`; `Inference()` được khởi tạo đúng với model đã tải và `window="whole"`. 171 → 174 test, `run_all.py -v` pass toàn bộ.

**Không phải lỗi chặn pipeline** (model không bị gate nên vẫn tải và chạy đúng), nhưng đáng sửa vì: (1) đúng cùng root cause đã biết là bug thật ở 9.7, chỉ chưa lan hết sang mọi điểm gọi `from_pretrained()` trong project; (2) nếu sau này HuggingFace đổi chính sách gate cho model này, hoặc bạn bị giới hạn tốc độ do request ẩn danh, token vẫn sẽ không được dùng dù đã cấu hình đúng trong `.env` - lỗi sẽ khó nhận ra vì không hề raise exception, chỉ có 1 dòng warning dễ bị bỏ qua giữa log dài. Bạn chạy lại `try_pipeline.py` (không cần thiết vì không có lỗi cần xác nhận hết) hoặc yên tâm bỏ qua bước xác nhận này - lần chạy tiếp theo sẽ không còn dòng cảnh báo "unauthenticated requests" đó nữa nếu muốn kiểm tra cho chắc.
