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
python -m src.main --collect 2008-01-01 2009-01-01 Go 100 10000 desc 
```

### Normalizing the Dataset

```bash
python -m src.main --normalize
```
### Analyze the Dataset

#### Distributions

```bash
python -m src.main --dist --variables stargazer_count --output ./output.png
```

#### Composite Scores

```bash
python -m src.main --composite --variables stargazer_count forks --name popularity 
```

#### Relationships

```bash 
python -m src.main --plot --variables stargazer_count forks --correlation pearson --output ./output.png

```

#### Heatmap

```bash
python -m src.main --heatmap --variables stargazer_count, forks, commit_count --correlation pearson --output ./output.png
```

---