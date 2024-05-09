# Greetings

Hello! I am Jari Haapasaari ([mail](mailto:haapjari@gmail.com)), and I originally developed the initial version of this project as a part of my thesis. Later I cleared and cleaned up the project, as it was pretty much just a collection of scripts. 

- Original Version Released: `20th October 2023`.
- Cleaner Version Released: `28th April 2024`.

If you're interested into reproduce the research, please see: [repository-analysis-orchestration](https://github.com/haapjari/repository-analysis-orchestration) repository.

## About

This is an `Interface` for tool for [repository-analysis-orchestration](https://github.com/haapjari/repository-analysis-orchestration) that offers a command-line interface for the tool. The tool is designed to collect, normalize, and display data from repositories. 

---

## TODO

- Fix Distribution, Hard to See
    - Left Skewed.
    - Add Rounded Corners.
- Clustering

---

## Example

### Display Help Message

```bash
python -m src.main --help
```

### Collecting the Dataset

```bash
python -m src.main --collect 2008-01-01 2009-03-01 Go 100 10000 desc 
```

### Normalizing the Dataset

```bash
python -m src.main --normalize
```

### Cleaning the Dataset

#### Drop a Column

```bash
python -m src.main --drop --table repos --column network_count
```

### Analyze the Dataset

#### Distributions

```bash
python -m src.main --dist --variables latest_release created_at stargazer_count open_issues closed_issues open_pull_request_count closed_pull_request_count forks watcher_count subscriber_count commit_count network_count total_releases_count contributor_count third_party_loc self_written_loc popularity activity maturity self_written_loc_proportion third_party_loc_proportion --output ./output.png
```

#### Create Weighted Sums

```bash
python -m src.main --composite --variables stargazer_count forks --name popularity 
```

#### Relationships

```bash 
python -m src.main --plot --variables stargazer_count forks --correlation pearson --output ./output.png

```

#### Heatmap

```bash
python -m src.main --heatmap --variables stargazer_count forks commit_count --correlation pearson --output ./output.png
```

#### Regression 

```bash
python -m src.main --regression --method linear --dependent stargazers --independent forks commits
```

#### Clustering

# TODO

---
