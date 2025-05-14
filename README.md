# NIS Coins Dataset

This repository contains a collection of high-resolution images of Israeli coins (New Israeli Shekel – NIS) for the purpose of labeling, training image classifiers, or visual documentation.

## 📁 Folder Structure & Contribution Guidelines

The main folder `images/` contains all coin images, organized by denomination.

If you'd like to contribute photos:

- Please create **your own subfolder** under `images/` using your name or identifier.
- Add only **clear, high-resolution images**.
- Label your images according to the following format:

### 🏷️ Coin Class Naming Convention:

| Coin Value | Class Name         |
|------------|--------------------|
| 1 Shekel   | `One`              |
| 2 Shekels  | `Two` *(or Shnekel)* |
| 5 Shekels  | `Five`             |
| 10 Shekels | `Ten`              |

Use these class names **exactly as written** to ensure consistent model training.

### 🔍 Unlabeled Images

- If you're unsure about the coin value, place the image in the `withoutlabels` subfolder.
- Others can help verify and move it to the correct category.

nis-coins/
├── images/             # Each contributor creates a subfolder here
│   ├── yiftach/
│   └──you... 
├── withoutlabels/      # For images without classification
└── README.md
