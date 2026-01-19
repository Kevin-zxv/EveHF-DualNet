# Project Workflow Guide

This project is based on the rpg_vid2e framework and contains the complete pipeline for processing video/images into event streams. Below are the detailed operational steps.

## Workflow Overview

1.  **Image Folder Adjustment**: Adjust the structure of original image folders.
2.  **Upsampling**: Generate upsampled files with counters.
3.  **Counter Matching**: Match generated counts files with event streams.
4.  **Event Stream Generation**: Use ESIM to generate event data.
5.  **Visualization**: Generate visual images of event streams.

---

## Detailed Steps

### 0. Image Folder Adjustment
Run the script to adjust the folder structure:
```bash
python adjust_image_folders.py
```

### 1. Generate Upsampled Files with Counters
Use `upsample_2.py` to upsample different datasets. Please modify the commands according to your actual input and output paths.

**Example Commands:**

*   **SGM-VFI Dataset**:
    ```bash
    python upsampling/upsample_2.py --input_dir="/root/autodl-tmp/sgm-vfi_25_60" \
                                  --output_dir="/root/autodl-tmp/upsampled_sgm-vfi_25_60"
    ```

*   **PerVFI Dataset**:
    ```bash
    python upsampling/upsample_2.py --input_dir="/root/autodl-tmp/PerVFI25_60/" \
                                  --output_dir="/root/autodl-tmp/upsampled_PerVFI25_60"
    ```

*   **UCF PerVFI Dataset**:
    ```bash
    python upsampling/upsample_2.py --input_dir="/root/autodl-tmp/UCF_PerVFI25/" \
                                  --output_dir="/root/autodl-tmp/upsampled_UCF_PerVFI25"
    ```

*   **Local Test Example**:
    ```bash
    python upsampling/upsample_2.py --input_dir="example/original/seq3" \
                                  --output_dir="example/upsampled/seq3"
    ```

### 2. Match Counts with Event Streams
Process the count files generated during upsampling to match them with the subsequent workflow.

```bash
# Standard Mode
python rename_and_delete_counts.py

# Or DANVIS Mode
python rename_and_delete_counts_DANVIS.py
```

### 3. Generate Event Streams
Use `esim_torch` to generate event data (`.npz` files).

**Example Commands:**

*   **Standard Generation**:
    ```bash
    python esim_torch/scripts/generate_events.py --input_dir="/root/autodl-tmp/upsampled_UCF_PerVFI25/" \
                                         --output_dir="/root/autodl-tmp/events" \
                                         --contrast_threshold_neg=0.2 \
                                         --contrast_threshold_pos=0.2 \
                                         --refractory_period_ns=0
    ```

*   **Local Test**:
    ```bash
    python esim_torch/scripts/generate_events.py --input_dir="example/upsampled" \
                                         --output_dir="example/events" \
                                         --contrast_threshold_neg=0.2 \
                                         --contrast_threshold_pos=0.2 \
                                         --refractory_period_ns=0
    ```

### 4. Generate Event Stream Images
Visualize the generated event stream data as images.

```bash
# Standard Visualization
python viz_events_3_2.py

# Or DANVIS Visualization
python viz_events_3_2_DANVIS.py
```


