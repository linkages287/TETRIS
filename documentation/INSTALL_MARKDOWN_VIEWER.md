# Install Markdown Viewer for Ubuntu

## Option 1: Glow (Recommended - Terminal-based, Beautiful)

**Install via pip (no sudo needed):**
```bash
pip3 install --user glow
```

**Add to PATH:**
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Or install system-wide (requires sudo):**
```bash
sudo apt-get update
sudo apt-get install -y glow
```

**Usage:**
```bash
glow ALGORITHM_DIAGRAM.md
```

---

## Option 2: Grip (Web-based Preview)

**Install via pip (no sudo needed):**
```bash
pip3 install --user grip
```

**Usage:**
```bash
python3 -m grip ALGORITHM_DIAGRAM.md
# Or if in PATH:
grip ALGORITHM_DIAGRAM.md
```

Then open `http://localhost:6419` in your browser.

---

## Option 3: mdcat (Terminal-based)

**Install via cargo (if Rust is installed):**
```bash
cargo install mdcat
```

**Or via package manager (if available):**
```bash
sudo apt-get install mdcat
```

**Usage:**
```bash
mdcat ALGORITHM_DIAGRAM.md
```

---

## Option 4: Pandoc (Convert to HTML/PDF)

**Install:**
```bash
sudo apt-get install pandoc
```

**Convert to HTML:**
```bash
pandoc ALGORITHM_DIAGRAM.md -o ALGORITHM_DIAGRAM.html
# Then open in browser:
xdg-open ALGORITHM_DIAGRAM.html
```

**Convert to PDF (requires texlive):**
```bash
sudo apt-get install texlive-latex-base
pandoc ALGORITHM_DIAGRAM.md -o ALGORITHM_DIAGRAM.pdf
```

---

## Option 5: GUI Markdown Editors

**ReText (Simple GUI editor/viewer):**
```bash
sudo apt-get install retext
```

**MarkText (Modern GUI editor):**
```bash
# Download from: https://github.com/marktext/marktext/releases
# Or via snap:
sudo snap install marktext
```

**Typora (Commercial, but excellent):**
```bash
# Download from: https://typora.io/
```

---

## Quick Test

After installing, test with:
```bash
cd /home/linkages/cursor/TETRIS
glow ALGORITHM_DIAGRAM.md
# OR
python3 -m grip ALGORITHM_DIAGRAM.md
# OR
pandoc ALGORITHM_DIAGRAM.md -o ALGORITHM_DIAGRAM.html && xdg-open ALGORITHM_DIAGRAM.html
```

---

## Recommendation

**For terminal viewing:** Use `glow` - it's beautiful and works great in terminal
**For web preview:** Use `grip` - renders GitHub-style markdown
**For conversion:** Use `pandoc` - convert to HTML/PDF for sharing

