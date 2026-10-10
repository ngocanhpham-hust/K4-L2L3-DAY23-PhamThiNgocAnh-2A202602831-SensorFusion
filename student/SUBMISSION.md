# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Phạm Thị Ngọc Anh
- MSSV: 2A202602831
- Email: 26ai.anhptn@vinuni.edu.vn
- Link repo (fork): https://github.com/ngocanhpham-hust/K4-L2L3-DAY23-PhamThiNgocAnh-2A202602831-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): Lấy hash 40 ký tự của commit cuối và nộp cùng URL trên VLearn.

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]`, `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `0`.
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: `0.970093`, `0.700405`, `519/16/222`.
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `0.150278 m`, `502`, `11.336915`, `0`, `239`, `2.522613`.
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `0.135848 m`, `502`, `9.264217`, `0`, `239`, `2.522613`.
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: Fused giảm RMSE khoảng `0.014430 m` so với LiDAR-only trong khi giữ nguyên `502` matches, `0` ghost và `239` miss. Vì số cặp và coverage không đổi, mức giảm RMSE phản ánh camera update đã tinh chỉnh vị trí của chính các track LiDAR, không phải đạt RMSE thấp bằng cách bỏ bớt cặp khó. Cả hai mode có `precision_track=1.0` và `coverage=502/519≈0.9672`. Tuy nhiên camera là tâm hộp FRONT ground-truth cộng nhiễu seed 0, không phải đầu ra camera detector, nên kết quả fused không chứng minh hiệu năng camera perception thực tế.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. LiDAR dùng `z=(x,y,z)` theo mét và `R=diag(sigma_lidar_x², sigma_lidar_y², sigma_lidar_z²)`; camera dùng `z=(u,v)` theo pixel và `R=diag(sigma_cam_i², sigma_cam_j²)`. LiDAR có mô hình đo tuyến tính vị trí, còn camera dùng phép chiếu pinhole phi tuyến trong `camera_measurement_prediction`; Jacobian `H` do platform tính tại state hiện tại.
2. Khoảng cách Mahalanobis chuẩn hóa innovation bằng `S=HPHᵀ+R`, nên xét cả độ bất định của track và sensor. Cổng χ² loại cặp không tương thích trước khi greedy assignment, giảm cập nhật nhầm ID và ngăn một outlier chiếm measurement của track đúng; camera ngoài FOV bị loại trước cả phép chiếu/Mahalanobis.
3. Đây là **track-then-fuse**: một danh sách track được predict đúng một lần mỗi frame, sau đó update LiDAR rồi mới update camera trên cùng state. Sau CP5, `grade_run.log` phải có đúng một record cho mỗi `(mode, frame)` của cả `lidar` và `fused`; các record `fused` là kết quả sau lượt camera, không phải hộp raw của hai sensor được gộp trước tracking.
4. Calibration sai làm dự báo pixel `h(x)` lệch có hệ thống, nên innovation `gamma=z-h(x)` có bias theo hướng/vị trí ảnh và khoảng cách Mahalanobis tăng. Nhiều cặp camera sẽ bị gate loại; nếu vẫn lọt gate, EKF có thể kéo state 3D lệch và làm RMSE/ghost/miss của fused xấu hơn.
5. `sensor` tường minh cho biết đây là lượt LiDAR hay camera ngay cả khi `meas_list=[]`. Lượt LiDAR rỗng vẫn phải xử lý miss trong FOV, giảm score và có thể xóa track; lượt camera rỗng phải giữ nguyên lifecycle. Theo thiết kế track-then-fuse, LiDAR có vị trí 3D nên được quyền tạo/score/xóa track, còn camera 2D chỉ tinh chỉnh state EKF của track đã có.
6. Hit LiDAR cộng `1/window` và chặn trên ở 1; track được confirmed chỉ khi `score > confirmed_threshold`. Miss không hạ state đã confirmed, nhưng trừ `1/window` nếu track ở trong LiDAR FOV. Xóa khi `P[0,0] > max_P` hoặc `P[1,1] > max_P`; hoặc confirmed có `score < delete_threshold`; hoặc track chưa confirmed có `score <= 0`.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không.

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): OpenAI Codex (ChatGPT).
- Dùng cho phần nào (hàm, câu hỏi, debug): Hỗ trợ cài đặt các hàm Part E–H, chuẩn bị môi trường Python 3.12, giải thích công thức EKF/association/lifecycle và debug ép scalar Mahalanobis trên NumPy 2.x.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Đối chiếu README/CHECKPOINTS/hướng dẫn kỹ thuật và công thức `F`, `Q`, innovation, `S`, Kalman gain, chi-square gating; chạy `pytest student/tests -q` được 128 passed; chạy Waymo `compare --seed 0` đủ frame 0–198 và dùng đúng số liệu do runner sinh trong `metrics.json`/log.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [ ] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
