# Greetings

Hello! I am Jari Haapasaari ([mail](mailto:haapjari@gmail.com)), and I originally developed the initial version of this project as a part of my thesis. Later I cleared and cleaned up the project, as it was pretty much just a collection of scripts. 

- Original Version Released: `20th October 2023`.
- Cleaner Version Released: `26th April 2024`.

If you're interested into reproduce the research, please see: [repository-analysis-orchestration](https://github.com/haapjari/repository-analysis-orchestration) repository.

## About

This is an `Interface` for tool for [repository-analysis-orchestration](https://github.com/haapjari/repository-analysis-orchestration) that offers a command-line interface for the tool. The tool is designed to collect, normalize, and display data from repositories. 

---

## Example

### Display Help Message

```bash
python -m src.main --help
```

### Collecting the Dataset

```bash
python -m src.main -c 2021-01-01 2021-01-08 Python 100 10000 desc 
```

### Normalizing the Dataset

```bash
python -m src.main --normalize
```
### Analyze the Dataset

#### Distributions

```bash
python -m src.main --analyze dist stargazers forks ./out/distributions.png
```

#### Relationships

```bash 
python -m src.main --analyze plot stargazers forks pearson ./out/relationship_plot.png
```

#### Heatmap

```bash
python -m src.main --analyze heatmap stargazers forks self_written_loc commit_count pearson ./out/heatmap.png
```

---