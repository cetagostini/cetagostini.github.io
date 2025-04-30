# Setting Up GitHub Pages for Your Quarto Website

Follow these steps to properly configure GitHub Pages for your Quarto website:

## 1. Create a GitHub Repository

If you haven't already:
1. Go to [GitHub](https://github.com) and create a new repository
2. Push your Quarto website code to this repository

## 2. Update Repository Settings

1. In your _quarto.yml file, replace the placeholders with your actual GitHub information:
   ```yaml
   website:
     site-url: https://YOUR-USERNAME.github.io/YOUR-REPOSITORY-NAME/
     repo-url: https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME
   ```

2. If your repository is named `YOUR-USERNAME.github.io`, use this configuration instead:
   ```yaml
   website:
     site-url: https://YOUR-USERNAME.github.io/
     repo-url: https://github.com/YOUR-USERNAME/YOUR-USERNAME.github.io
   ```

## 3. Configure GitHub Pages

1. Go to your repository on GitHub
2. Click on "Settings" (tab at the top)
3. Navigate to "Pages" (in the sidebar)
4. Configure your source settings:
   - Under "Source", select "GitHub Actions"

## 4. Verify Workflow File

The `.github/workflows/quarto-publish.yml` file has already been created for you. It will:
- Install Quarto
- Render your website
- Deploy the rendered site to GitHub Pages

## 5. Trigger the Deployment

To trigger the deployment:
1. Push a commit to the `main` branch
2. Or manually run the workflow from the "Actions" tab on GitHub

## 6. Verify Deployment

1. Go to the "Actions" tab in your repository
2. You should see the workflow running (or completed)
3. If successful, your site will be available at your site-url (e.g., https://YOUR-USERNAME.github.io/YOUR-REPOSITORY-NAME/)

## Troubleshooting

If you encounter any issues:
1. Check the workflow run logs in the Actions tab
2. Make sure the `_quarto.yml` file has the correct site-url and repo-url values
3. Ensure the .nojekyll file exists in your repository
4. Verify that GitHub Pages is enabled in your repository settings

## Additional Resources

- [Quarto Publishing with GitHub Pages](https://quarto.org/docs/publishing/github-pages.html)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions) 