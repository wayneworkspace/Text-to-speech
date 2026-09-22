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
