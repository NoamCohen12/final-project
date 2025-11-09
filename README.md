# Advanced Quantization Methods for Accelerating Machine Learning

This repository contains our final-year research project in Computer Science.  
The work was performed by a team of 3 researchers as part of our B.Sc. capstone, under academic supervision by **Dr. Itamar Cohen**.

## Motivation

Quantization is a critical direction in accelerating inference of deep neural networks — reducing memory, reducing compute cost, and enabling deployment on constrained edge hardware.

In this project we systematically investigated **which combination** of:
- bit-width  
- quantization method  
- and clamping strategy  

produces the best tradeoff between accuracy degradation and compression speedup.

This is a **real applied research** investigation — not a coding exercise — intended to explore directions relevant for industrial deployment.

## Scope Summary

We examined and compared:

| Dimension | Variants Explored          |
|----------|----------------------------|
| Representations | INT8, INT16                |
| Schemes | symmetric, asymmetric      |
| With/without clamping | multiple clipping thresholds |
| Comparison vs F2P novel numerical representation | from prior research phase  |

The main model evaluated was **ResNet18**, using controlled repeatable evaluation over the same dataset.

## Code Structure

| Folder | Purpose |
|--------|---------|
| `src/` | main quantization engine + pipeline implementation |
| `tests/` | unit tests + vector sanity experiments |
| `experiments/` | full model quantization experiment harnesses |
| `demos/` | demo visualizations + tensor behavior exploration |
| `data/` | dataset and imagenet class index labels |
| `reports/` | generated experiment result text files |
| `test_results/` | baseline comparison logs (F2P vs INT) |
| `vector_output_one_layer/` | raw per-layer quantization outputs |

The most important file is:

```
src/full_quantization_pipeline.py
```

This is the entry point.  
Inside main() you can switch between multiple configuration presets.

## Running

Just run:

```bash
python src/full_quantization_pipeline.py
```

Then choose which configuration you want to activate inside the code (comment / uncomment groups inside main).

## Results + Paper

- Final evaluated graphs, methodology, results and conclusions are in:  
**Final_Project_Report.pdf**

- Raw numeric experiment datasets are in:  
**result_of_quantization_final.docx**

## Presentation

A summarized version for high-level overview appears in:  
**Final_Project_Presentation.pdf**

## Notes

This repository intentionally keeps many experiment permutations — because the goal of this research was **exploration** — not a single pipeline.  
Recruiters and technical reviewers can see the progression and depth of experimentation.


