# Marketing Science Blog

This is a [Quarto](https://quarto.org) website for sharing articles, talks, and information about marketing science.

## Local Development

### Prerequisites

- [Quarto](https://quarto.org/docs/get-started/) (version 1.3.0+)
- [R](https://cran.r-project.org/) (optional, only if using R code)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/USERNAME/REPOSITORY-NAME.git
   cd REPOSITORY-NAME
   ```

2. Preview the website locally:
   ```bash
   quarto preview
   ```

3. Build the website:
   ```bash
   quarto render
   ```

## GitHub Pages Deployment

This website is automatically deployed to GitHub Pages whenever changes are pushed to the `main` branch using GitHub Actions.

### Manual Deployment

If you need to deploy manually:

1. Build the site:
   ```bash
   quarto render
   ```

2. Push the changes to GitHub:
   ```bash
   git add .
   git commit -m "Update website"
   git push
   ```

## Site Structure

- `index.qmd`: Home page
- `about.qmd`: About page with professional experience
- `articles.qmd`: Articles listing
- `talks.qmd`: Talks and presentations
- `styles.css`: Custom styling
- `_quarto.yml`: Site configuration
- `js/`: JavaScript files
- `images/`: Image assets

## Customization

1. Edit the `_quarto.yml` file to update site metadata and navigation
2. Modify the content in the .qmd files to update page content
3. Update the `styles.css` file to change the visual styling

## License

This project is licensed under the MIT License - see the LICENSE file for details. 