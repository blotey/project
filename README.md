# 🧠 Image Captioning + Segmentation App (Deep Learning)

This project combines:
- **Image Captioning** (Transformer-based) 📸
- **Semantic Image Segmentation** (U-Net-based) 🧩

All wrapped in a simple **Streamlit Web App**.

---

## 💡 Features

- Upload a single image
- View generated image caption
- View predicted segmentation mask overlay
- Downloadable outputs

---

## 📁 Folder Structure

```bash
project/
├── app/              # Streamlit UI
├── captioning/       # Transformer model + training
├── segmentation/     # U-Net model + training
├── data/             # Image, caption, and mask datasets
├── output/           # Logs and predicted masks
