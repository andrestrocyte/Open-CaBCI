# Open-CaBCI

**A real-time, closed-loop two-photon calcium imaging brain-computer interface for volitional ensemble control.**

---

## Overview

Open-CaBCI is an open-source software platform for running closed-loop BCI experiments using two-photon calcium imaging in head-fixed mice. It enables real-time recording, online ensemble-state computation, auditory feedback delivery, and reward control, with a graphical interface for live session monitoring and control.

---

## Associated Publication

**Equivalent volitional learning emerges through circuit-specific population dynamics in motor cortex and hippocampus**

Andres de Vicente\*, Catalin Mitelut\*, Renan V. Mendes, Lorenzo Marianelli, Mariona Colomer Rosell, David Bruckner, Giampiero Bardella, and Flavio Donato

Biozentrum, University of Basel | Sapienza, University of Rome

\*Equal contribution | Correspondence: flavio.donato@unibas.ch

> bioRxiv DOI: *coming soon*

---

## Paradigm Overview

![BCI Paradigm](figures/paradigm_overview.png)

**(A)** Schematic of the closed-loop two-photon calcium imaging BCI paradigm. Head-fixed mice were trained to volitionally modulate selected neuronal ensembles in M1 or CA3. The protocol comprised an ensemble-selection session (day 0) followed by 8 daily training sessions, each with a 15-min calibration period and a 50-min BCI session.

**(B)** Example field of view and task structure, including ensemble composition, calcium traces, ensemble state computation, auditory cursor feedback, and trial structure.

**(C)** Real-time graphical user interface from Open-CaBCI, used to monitor and control task performance during BCI sessions, including session metrics, field-of-view display, motion correction, and reward threshold control.

---

## Requirements

- Python 3.x
- Dependencies: see `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## Usage

Please refer to the documentation in the repository for detailed usage instructions.

---

## Repository Structure

```
Open-CaBCI/
├── figures/          # Figures for documentation
├── src/              # Core source code
├── requirements.txt  # Dependencies
└── README.md
```

---

## Credits

This repository builds on tools originally developed by **Catalin Mitelut** ([@catubc](https://github.com/catubc/bmi_tools)). We are grateful for his foundational contributions to this work.

---

## License

This project is licensed under the [GPL-3.0 License](https://github.com/donatolab/Open-CaBCI#GPL-3.0-1-ov-file).

---

## Contact

**Donato Lab** | Biozentrum, University of Basel
🌐 [donatolab.com](https://www.donatolab.com) | ✉️ flavio.donato@unibas.ch
