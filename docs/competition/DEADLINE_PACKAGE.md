# VAIIF26 Deadline Package

This package is written for the August 25, 2026 submission. It deliberately does not claim clinician endorsement, hospital validation, diagnostic accuracy, or physical Intel-hardware validation.

## Submission-ready project title

**Pulse-OXR: Offline Grounded Chest X-Ray Copilot**

Word count: 6 (limit: 10).

## Submission-ready project description

Pulse-OXR is a local-first research prototype designed to help clinicians review chest X-rays where radiology expertise or internet access may be limited. It accepts a DICOM image, clears configured identifying fields in memory, and combines three pretrained AI stages: BiomedCLIP for open-vocabulary screening, MedSAM for experimental candidate regions, and CheXagent or MedGemma for a draft report. A hardware-aware launcher selects Apple MLX, NVIDIA CUDA, AMD ROCm, Intel XPU, or CPU execution and chooses quantization from available memory. The interface supports English and Vietnamese, exposes candidate-region evidence, warns when a positive statement has no matching region, records only privacy-preserving edit statistics, and requires physician review. Pulse-OXR does not claim diagnostic accuracy or clinical validation. Current evidence consists of automated tests and controlled, de-identified software demonstrations. The next phase is licensed-clinician evaluation and prospective validation before any clinical use.

Maximum permitted length: 150 words. Recount the text in the final registration form because some forms treat hyphenated terms differently.

### Vietnamese reference translation

Pulse-OXR là nguyên mẫu nghiên cứu ưu tiên xử lý cục bộ, nhằm hỗ trợ nhân viên y tế xem xét ảnh X-quang ngực tại những nơi khả năng tiếp cận bác sĩ chẩn đoán hình ảnh hoặc Internet còn hạn chế. Hệ thống nhận một ảnh DICOM, xóa các trường định danh đã cấu hình trong bộ nhớ, rồi kết hợp ba tầng AI được huấn luyện sẵn: BiomedCLIP để sàng lọc mở, MedSAM để tạo vùng ứng viên thử nghiệm, và CheXagent hoặc MedGemma để soạn bản nháp báo cáo. Trình triển khai tự nhận diện phần cứng, chọn Apple MLX, NVIDIA CUDA, AMD ROCm, Intel XPU hoặc CPU, đồng thời chọn mức lượng tử hóa theo bộ nhớ khả dụng. Giao diện hỗ trợ tiếng Anh và tiếng Việt, hiển thị trạng thái bằng chứng và luôn yêu cầu bác sĩ xem xét. Pulse-OXR chưa tuyên bố độ chính xác chẩn đoán hoặc xác thực lâm sàng.

## Two-minute flagship video

### Locked production format

- Final master: 16:9, 1920×1080, 25 fps, 1:59 target duration.
- Spoken language: English. Burn in the supplied Vietnamese subtitle track.
- A-camera: Sony a6000 on the stable outdoor table. Use it for every spoken line.
- B-camera: iPhone 14 Pro in Blackmagic Camera. Use it for brief cutaways and over-the-shoulder shots, not as the only recording of a spoken line.
- Screen source: a native screen recording of the app. Do not film the laptop display for the readable demo shots.
- Speaking order: Thịnh opens and explains deployment; Hiếu explains the pipeline and safety; Khang leads the demo and closes.
- Narration length: approximately 231 English words. Rehearse at a calm conversational pace; do not speed-read to recover a long take.

Use a visible jump cut over model loading and show the caption **“Inference wait shortened for this two-minute video.”** Do not imply that the edited wait is real-time inference.

### Timed three-person script

| Time | Speaker and picture | Exact English narration |
| --- | --- | --- |
| 0:00-0:17 | **Thịnh on camera.** Begin on a stable medium three-shot, then cut tighter to Thịnh. Add lower thirds for all three names and roles. | Chest X-rays are essential, but fast radiology support is not equally available. We are Pulse-OXR: Thịnh, team lead; Hiếu, software engineer; and Khang, co-software engineer. We built a local-first research copilot for resource-constrained care. |
| 0:17-0:33 | **Hiếu on camera** for the first sentence; cover the model names with a clean three-stage pipeline graphic or the app’s pipeline panel. | It does not replace a doctor. BiomedCLIP screens image tiles, MedSAM proposes candidate regions, and MedGemma or CheXagent drafts findings. The system runs locally, strips configured identifiers in memory, and keeps uncertainty visible. |
| 0:33-1:09 | **Khang on camera** for “Here is the workflow,” then native app screen recording. Show upload, model selection, Run, the disclosed loading jump cut, viewer, overlays, and evidence audit in that order. | Here is the workflow. I upload a de-identified demonstration DICOM, select MedGemma, and run analysis. The viewer shows the decoded radiograph and any input-quality warnings. I can toggle experimental regions, then inspect each report statement in the evidence audit. A positive claim without a matching region is marked text-only and unverified. A missing overlay never proves a normal result; it only means spatial support was not found. |
| 1:09-1:26 | **Hiếu on camera** for the opening clause, then screen close-ups of a changed upload, editable report, and privacy text. | We also clear stale results when the file changes, delete temporary uploads, and store only edit counts and one-way hashes—not report text—in the local feedback log. Every output remains editable and requires physician review. |
| 1:26-1:44 | **Thịnh on camera** beside the table-supported laptop; cover framework names with the runtime panel and one-click command. | Our one-click deployer detects memory and chooses Apple MLX, NVIDIA CUDA, AMD ROCm, Intel XPU, or CPU with suitable quantization. This demonstration runs on a 24-gigabyte M4 Pro; Intel execution is implemented but not yet hardware-validated. |
| 1:44-1:59 | **Khang on camera**, centered, with Thịnh and Hiếu in the wider frame behind or beside him. End on the team and project name. | Our evidence today is controlled software testing, not clinical validation. Next comes licensed-clinician evaluation. Pulse-OXR supports SDG 3 through transparent, offline-first assistance that keeps people—not AI—in control. |
| 1:59-2:00 | One-second end card. | No narration. Show “Research prototype · Physician review required.” |

### Vietnamese subtitle track

Import [`VIDEO_SUBTITLES_VI.srt`](VIDEO_SUBTITLES_VI.srt) into a dedicated subtitle track. It is split into short cues rather than six paragraph-sized captions. After the English edit is locked, retime individual cues to the actual spoken words without changing their meaning.

Use a clean sans-serif face, white text, and a dark semi-opaque box or strong shadow. Keep subtitles to two lines, inside title-safe margins, and above lower thirds. Render them into the submitted master unless the form explicitly accepts a separate subtitle file.

## Field production plan for the available equipment

### Camera configuration

| Setting | Sony a6000 + Zeiss 16-70mm f/4 (A-camera) | iPhone 14 Pro + Blackmagic Camera (B-camera) |
| --- | --- | --- |
| Role | All dialogue and the master image | Silent cutaways, alternate angle, hands, laptop, group, environment |
| Format | PAL **25p**, 1920×1080. Prefer XAVC S at the highest available 25p bitrate. If XAVC S is unavailable, use AVCHD 25p 24M and test one clip in Resolve before the full shoot. | **4K 25 fps, H.264, High or Max bitrate, Rec.709/SDR.** This gives reframing room while remaining friendly to Resolve Free. Do not use HDR, Apple Log, Cinematic mode, or ProRes for this deadline shoot. |
| Shutter | Manual exposure, **1/50 s** | **180°** (equivalent to 1/50 s at 25 fps) while exposure permits |
| Exposure | Start at ISO 100. Use aperture to expose; f/4-f/5.6 for a single speaker and f/5.6-f/8 for the group when light allows. Protect faces and bright shirts with zebra at 100/100+. | Set ISO as low as the app permits and watch the histogram/zebras. If highlights clip at 180° without an ND filter, move deeper into shade first. Only then use a smaller shutter angle/faster shutter; keep those shots mostly static. |
| White balance | Make a Custom Setup from the same neutral white/grey reference under the actual speaking light, then keep it locked. | Frame the same reference, set or auto-measure white balance once, then lock white balance and tint. Do not leave auto white balance running. |
| Focus | For a seated/standing mark, focus on the eyes, switch to manual focus, magnify to confirm, and tape/mark the standing position. If speakers move, use Continuous AF with normal tracking and test for hunting. | Use the 1× main rear camera. Lock focus on the speaker or laptop before each shot; avoid digital zoom. |
| Framing | About 24-35mm on the lens for singles; 16-24mm for the group if space is tight. Keep the camera level and near eye height. | Landscape orientation. Brace with two hands, against the table, or against a fixed object. Make only slow, short movements. |

The a6000 gained XAVC S through firmware version 2.00 and later, and Sony requires a suitably formatted Class 10-or-faster SDXC card for that mode. Do not perform a firmware update at the location; use the AVCHD fallback if the menu is absent.

### Light and location strategy

1. Use bright open shade near the edge of a building or large tree, not direct sun and not patchy leaf shadows. Keep the background in similar or lower brightness than the faces.
2. Put the sun behind and slightly to one side of the team. Do not face everyone into the sun; squinting and hard eye shadows are difficult to repair.
3. Because there is no ND filter, solve overexposure by moving deeper into shade and closing the Sony aperture. Do not let either camera clip faces simply to preserve 1/50 s.
4. Record all spoken segments before decorative B-roll so changing late-afternoon light does not create continuity jumps. Re-record the neutral reference if the light changes visibly.
5. Stabilize the improvised table: use a folded cloth under the camera if needed, weigh the table or camera base with a bag, disable contact with the table during takes, and inspect the horizon before every setup.

### Phone-recorder audio procedure

Outdoor audio is the highest-risk part of this shoot. Post-processing cannot restore words destroyed by wind or clipping.

1. Put each recorder phone in Airplane Mode and Do Not Disturb. Choose its highest-quality or lossless setting and 48 kHz if the recorder offers it.
2. Record one speaker at a time. A teammate should hold the phone 20-30 cm below the speaker's chin, just outside the Sony frame, with the microphone opening unobstructed. Do not put it loose in clothing where fabric can rub it.
3. Use the second available phone as a backup recorder on the speaker's other side. The Sony and iPhone camera audio are scratch tracks for synchronization, not the preferred dialogue source.
4. After every device is rolling, say the speaker, segment, and take number, then make one sharp hand clap in view of both cameras. Leave two seconds of silence before the line.
5. Stand on the sheltered side of a wall, hedge, or dense tree line and keep the recorder on the leeward side of the speaker. Record and play back a ten-second wind test before the first real take.
6. Capture 30 seconds of location ambience with nobody talking. Record at least two complete good takes of each segment. Redo any take with a gust, handling noise, missed word, or clipped syllable immediately.

### Minimum shot order

1. Native screen recording: pre-warm the model, rehearse the exact click path, and capture one clean successful run plus static holds on the evidence and runtime panels.
2. Calibration: five seconds of the same neutral reference on both cameras, followed by the audio/clap synchronization test and immediate playback.
3. Thịnh: record segments 1 and 5 twice each.
4. Hiếu: record segments 2 and 4 twice each.
5. Khang: record segments 3 and 6 twice each; keep the complete spoken demo take even though much of it will sit under screen footage.
6. Group opening and closing holds, then only the B-roll named in the timed script.
7. Before leaving, verify picture, focus, and close-phone audio for every required segment on the actual files, not only on camera thumbnails.

## DaVinci Resolve Free workflow

### 1. Project and media setup

1. Before making the timeline, set **1920×1080, 25 fps**, and **48 kHz** project audio. Resolve locks important frame-rate choices after timeline creation.
2. Keep the project in an SDR Rec.709 workflow: DaVinci YRGB, timeline/output color space Rec.709 Gamma 2.4. The chosen sources are already non-Log SDR, so do not apply an Apple Log LUT or a Color Space Transform.
3. Create bins named `A_CAM_SONY`, `B_CAM_IPHONE`, `PHONE_AUDIO`, `SCREEN`, `GRAPHICS`, and `EXPORTS`. Rename clips by speaker, segment, and take before editing.
4. Synchronize each camera take with its close-phone recording by waveform. If automatic sync misses, line up the visible clap with the sharp audio spike. Keep the camera scratch track muted but available for checking sync drift.

### 2. Edit for a truthful 1:59

1. Build the dialogue-only spine first in script order. Select the best performance, not a patchwork of every sentence unless needed.
2. Cover cuts with the native screen recording and short B-camera shots. Prefer straight cuts and restrained J/L cuts; avoid transitions that spend time without adding information.
3. Keep the upload, model choice, Run click, disclosed loading jump, decoded image, evidence audit, and runtime panel visible long enough to understand. Do not speed through the safety labels.
4. Show the loading disclosure for at least two seconds. The final disclaimer card receives the last second.

### 3. Match the two non-Log cameras

1. Choose the best Sony close-up as the visual reference and grab a still in the Color page.
2. On every clip, use one node for exposure/white balance and a second for contrast/saturation. Make small corrections; both selected recording paths are intended for a direct Rec.709 finish, not aggressive recovery.
3. Use waveform and RGB parade to remove a color cast and prevent clipped face highlights. Use the vectorscope skin-tone indicator as a direction, then judge the actual faces rather than forcing every complexion to one brightness value.
4. Match iPhone shots to the Sony reference with split-screen or still wipe: white balance first, exposure second, contrast third, saturation last. Copy the base correction only between shots recorded under the same light.
5. Do not add a “cinematic” LUT, film grain, heavy teal/orange treatment, or sharpening. The demo UI and subtitles need neutral, consistent color more than a stylized look.

### 4. Clean dialogue with Free tools

1. Use clip gain first so normal speech lands consistently. As a practical web-video target, keep final dialogue peaks below -1 dBFS and check the complete mix around -14 to -16 LUFS integrated; this is a delivery target, not a competition rule.
2. In Fairlight, use the channel EQ for a gentle high-pass around 70-90 Hz to reduce handling and wind rumble. Do not raise the cutoff until voices sound thin.
3. Use light compression, approximately 2:1 with only 3-6 dB of gain reduction on louder words, followed by a limiter at -1 dB. Add de-essing or hum removal only when the recording actually needs it.
4. Apply short crossfades at dialogue edits and use the captured ambience underneath to prevent dead-silent cuts.
5. Resolve Studio's AI voice isolation and temporal/spatial video noise reduction are not part of this Free workflow. If wind masks a word, use the backup phone or another take rather than trying to manufacture intelligibility in post.

### 5. Subtitles, graphics, and delivery

1. Import `VIDEO_SUBTITLES_VI.srt` as a subtitle track, retime it after picture lock, then review every Vietnamese line against the English audio.
2. Use lower thirds exactly once per person: `THỊNH — TEAM LEAD`, `HIẾU — SOFTWARE ENGINEER`, and `KHANG — CO-SOFTWARE ENGINEER · DEMO PRESENTER`.
3. Export a 1920×1080, 25 fps MP4 using H.264 at roughly 15 Mb/s with AAC stereo at 48 kHz and 320 kb/s. If the registration form reveals a smaller file limit, reduce bitrate rather than frame rate or resolution.
4. Watch the rendered file from beginning to end in a normal web browser, once with sound and once muted. Confirm duration, subtitle safety, audio sync, readable UI text, no patient identifiers, and no unsupported validation claim.

### Manufacturer references used for this plan

- [Sony a6000 firmware and XAVC S requirements](https://www.sony.com/electronics/support/e-mount-body-ilce-6000-series/ilce-6000/downloads/00015954)
- [Sony a6000 25p recording settings](https://helpguide.sony.net/gbmig/45349331/v1/en/contents/TP0000518238.html)
- [Sony a6000 custom white balance](https://helpguide.sony.net/gbmig/45349331/v1/en/contents/TP0000518345.html)
- [Sony a6000 zebra exposure guide](https://helpguide.sony.net/gbmig/45349331/v1/en/contents/TP0000508219.html)
- [Apple iPhone 14 Pro video specifications](https://support.apple.com/en-us/111849)
- [Blackmagic Camera controls and recording overview](https://www.blackmagicdesign.com/products/blackmagiccamera)
- [DaVinci Resolve official training](https://www.blackmagicdesign.com/products/davinciresolve/training)
- [DaVinci Resolve subtitles and closed captions](https://www.blackmagicdesign.com/products/davinciresolve/edit)
- [DaVinci Resolve Free and Studio feature differences](https://www.blackmagicdesign.com/products/davinciresolve/studio)

## Recording checklist

1. Run the offline preflight and save its terminal result:

   ```bash
   ./scripts/preflight.py --dicom ~/Documents/dicom_tests/00001075_000_secondary_capture.dcm
   ```

2. Start the UI with `./scripts/run_ui.sh` and run the demo once before recording so model initialization is complete.
3. Use only the de-identified secondary-capture demonstration DICOM. Keep the UI’s secondary-capture warning visible or briefly expand it.
4. Set the browser zoom so the viewer, report, evidence audit, and runtime details are readable.
5. Record the screen sections separately from the face-to-camera sections, then edit to the 1:59 target without exceeding two minutes.
6. Add the supplied Vietnamese subtitle track, project title, all three names and roles, and the non-diagnostic disclaimer.
7. Listen once with the screen hidden: the narration must still explain problem, audience, AI, safety, hardware, present evidence, and next step.
8. Watch once muted: captions and screen actions must still tell the story.

## What is operational now

- One-click hardware and memory detection with framework/quantization selection.
- DICOM decoding, in-memory clearing of configured PHI fields, temporary-file cleanup, and image preview.
- Pretrained BiomedCLIP, MedSAM, and MedGemma/CheXagent pipeline; no custom training claim.
- Experimental candidate-region overlays and per-claim evidence status.
- English/Vietnamese interface labels and safety text.
- Editable research draft, privacy-preserving edit statistics, runtime timing, and model manifest.
- Automated regression tests and an offline preflight command.

## Claims that must remain explicit limitations

- No clinician, hospital, or prospective clinical validation has been completed.
- No sensitivity, specificity, AUROC, or patient-outcome claim is supported.
- The NIH demonstration image checks software operation only; it is not evidence of diagnostic accuracy and is not claimed as MedGemma training data.
- Candidate regions are generated proposals, not ground-truth lesion boundaries.
- BiomedCLIP prompt scores are relative zero-shot scores, not calibrated disease probabilities.
- Physical Intel XPU execution has not yet been demonstrated; an offline hardware-profile simulation is not hardware validation.
- The converted secondary-capture DICOM is appropriate for the software demo, not acquisition-fidelity evaluation.

## Tomorrow-only priority order

1. Green preflight and clean full test suite.
2. One successful MedGemma demo run using the de-identified DICOM.
3. Record and edit the two-minute video.
4. Paste the title and description, verify their form word counts, upload the signed consent, and submit.
5. Only if time remains: make the repository public after a credential/data/license review and record the final commit URL.

Do not spend the remaining window on model training, a fake accuracy evaluation, cold outreach for rushed clinical endorsement, or an unverified Intel performance claim.

## Evidence sources for the problem statement

- [The Growing Problem of Radiologist Shortage: Vietnam's Perspectives](https://pubmed.ncbi.nlm.nih.gov/37899516/)
- [World Bank evaluation of AI-powered chest radiography at Thanh An Commune Health Station](https://documents1.worldbank.org/curated/en/099061225041027465/pdf/P181325-256e34ed-bd6d-470c-8c2f-3164c41a44a8.pdf)
- [Official Intel Vietnam AI Impact Festival 2026 announcement](https://shtp.hochiminhcity.gov.vn/en/intel-vietnam-ai-impact-festival-2026-launchpad-for-the-next-generation-of-young-talents-940.htm)
