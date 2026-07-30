---
title: Evolution Trainer
emoji: 🦿
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Evolution trainer

Headless [genetic-algorithm creature trainer](https://github.com/keiwando/evolution)
running on the free CPU tier. Trains continuously in the background and pushes
`best.npz` / `history.csv` / `config.json` to a separate Hugging Face repo every
few minutes so progress survives even if this Space sleeps or restarts.

Pull the latest checkpoint from your own machine:

```bash
python main.py sync --from hf://<push-repo> --run runs/arch --interval 60
python main.py watch --run runs/arch --live
```

This page itself is a read-only listing of the run directory (served by
`main.py serve`), mostly useful for confirming the container is alive.
