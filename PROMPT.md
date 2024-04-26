Could you alter my "Collect" interface, to allow users to search for repositories based on the following criteria:

- First Creation Date
- Last Creation Date
- Language
- Minimum Stars
- Maximum Stars
- Order

If the first and last creation date are more than a week apart, then the query will be split to multiple queries, where single query will be for a week.

I'd like to have:

- Adjusted Python Main Function to Support Arguments to "Collect" Flag.
- Adjusted Example Script to Demonstrate the Usage of "Collect" Flag.

Here is the example codes:

Backend API:

```yaml
paths:
  /api/v1/repos/search:
    get:
      summary: Abstraction of GitHub Search API.
      description: Abstraction of GitHub Search API.
      externalDocs:
        description: GitHub Repository Search API Docs
        url: https://docs.github.com/en/rest/search/search?apiVersion=2022-11-28#search-repositories
      parameters:
        - in: query
          name: firstCreationDate
          schema:
            type: string
          required: true
          description: YYYY-MM-DD
          example: "2013-05-01"
        - in: query
          name: lastCreationDate
          schema:
            type: string
          required: true
          description: YYYY-MM-DD
          example: "2013-05-01"
        - in: query
          name: language
          schema:
            type: string
          required: true
          example: Go
        - in: query
          name: minStars
          schema:
            type: string
          required: true
          description: Minimum Stars repository must have.
          example: "100"
        - in: query
          name: maxStars
          schema:
            type: string
          required: true
          description: Max Stars repository must have. If set to 0, it will be considered as no limit.
          example: "10000"
        - in: query
          name: order
          schema:
            type: string
            enum: [ asc, desc ]
            default: desc
          required: false
          description: The order of the results, either ascending (asc) or descending (desc). Defaults to descending.
          example: desc
      responses:
        '200':
          description: Successful
          content:
            application/json:
              schema:
                type: object
                properties:
                  total_count:
                    type: integer
                    description: The total number of repositories found.
                  items:
                    type: array
                    items:
                      $ref: '#/components/schemas/Repository'
                required:
                  - total_count
                  - items
        '400':
          description: Bad Request
          content:
            application/json:
              schema:
                type: object
                properties:
                  error:
                    type: string
                    description: Error Message.
        '403':
          description: Forbidden
          content:
            application/json:
              schema:
                type: object
                properties:
                  error:
                    type: string
                    description: Error Message.
        '500':
          description: Internal Server Error
          content:
            application/json:
              schema:
                type: object
                properties:
                  error:
                    type: string
                    description: Error Message.
        '503':
          description: Service Unavailable
          content:
            application/json:
              schema:
                type: object
                properties:
                  error:
                    type: string
                    description: Error Message.
```

```python
def main():
    log.basicConfig(level=log.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: [OPTIONS] [ARGUMENTS]")
        print("")

        print("Options:")
        print("  -h, --help\t\t\tDisplay this help message")
        print("  -c, --collect\t\t\tCollect dataset and save it to the database")
        print("  -n, --normalize\t\tNormalize and clean dataset")
        print("  -a, --analyze [action] [var1] [var2] [correlation] [output_path]")
        print("")

        print("Arguments for --analyze:")
        print("  action\t\t\tSpecify the analysis type: 'dist', 'plot', 'heatmap'")
        print("  var1\t\t\t\tFirst variable for analysis (optional for heatmap)")
        print("  var2\t\t\t\tSecond variable for analysis (optional for heatmap)")
        print("  correlation\t\t\tCorrelation method: 'spearman', 'kendall', 'pearson' (optional, defaults to 'pearson')")
        print("  output_path\t\t\tPath to save the analysis output picture")
        print("")

        print("Actions:")
        print("  dist\t\t\t\tDraw distributions for given variables")
        print("  plot\t\t\t\tDraw relationship plots for given variables using specified correlation method")
        print("  heatmap\t\t\tDraw a heatmap for given variables using specified correlation method")
        print("")

        print("Examples:")
        print("  python main.py --collect")
        print("  python main.py --normalize")
        print("  python main.py --analyze dist stargazers forks out/distributions.png")
        print("  python main.py --analyze plot stargazers forks pearson out/relationship_plot.png")
        print("  python main.py --analyze heatmap stargazers forks spearman out/heatmap.png")

        return

    if sys.argv[1] in ("-c", "--collect"):
        log.debug("Collecting dataset...")
        # TODO: Implementation of collection logic

    elif sys.argv[1] in ("-n", "--normalize"):
        log.debug("Normalizing dataset...")
        # TODO: Implementation of normalization logic

    elif sys.argv[1] in ("-a", "--analyze"):
        if len(sys.argv) < 5:
            print("Insufficient arguments. See '--help'.")
            return

        action = sys.argv[2]
        output_path = sys.argv[-1]
        variables = sys.argv[3:-2]  # Excludes the correlation method and output path
        correlation = sys.argv[-2] if action in ['plot', 'heatmap'] and sys.argv[-2] in ['spearman', 'kendall', 'pearson'] else 'pearson'

        if action == "dist":
            log.debug(f"Drawing distributions for variables {', '.join(variables)} and saving to {output_path}")
            # TODO: Distribution drawing logic

        elif action == "plot":
            log.debug(f"Drawing relationship plots for variables {', '.join(variables)} using {correlation} correlation and saving to {output_path}")
            # TODO: Relationship plot drawing logic

        elif action == "heatmap":
            log.debug(f"Drawing heatmap for variables {', '.join(variables)} using {correlation} correlation and saving to {output_path}")
            # TODO: Heatmap drawing logic

    else:
        print("Unknown option. See '--help' for more information.")
```

### Collecting the Dataset

```bash
python main.py --collect 
```