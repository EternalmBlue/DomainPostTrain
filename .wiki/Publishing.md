# Publishing

Use this page when publishing `.wiki/` source files to the real GitHub Wiki.

GitHub Wikis are Git repositories. Creating `.wiki/` in this main repository does not automatically make pages live on GitHub.

## GitHub Wiki mechanics

Current GitHub Docs state that:

- Wiki pages can be edited on GitHub or locally.
- A Wiki can be cloned after an initial page exists.
- The clone URL is shaped like `https://github.com/OWNER/REPO.wiki.git`.
- Only changes pushed to the Wiki default branch are live to readers.
- Filenames determine page titles.
- File extensions determine rendering.
- `_Sidebar.md` is used as the custom sidebar.
- Avoid these filename characters: `\ / : * ? " < > |`.

Official references:

- https://docs.github.com/en/communities/documenting-your-project-with-wikis/adding-or-editing-wiki-pages
- https://docs.github.com/en/communities/documenting-your-project-with-wikis/creating-a-footer-or-sidebar-for-your-wiki
- https://docs.github.com/en/communities/documenting-your-project-with-wikis/changing-access-permissions-for-wikis

## One-time setup

1. In GitHub, open the repository.
2. Open the Wiki tab.
3. Create an initial page if the Wiki is empty.
4. Clone the Wiki repository:

```bash
git clone https://github.com/OWNER/REPO.wiki.git
```

Replace `OWNER` and `REPO` with the real repository owner and name.

## Publish from this source directory

From the main repository root:

```bash
cp .wiki/*.md ../REPO.wiki/
cd ../REPO.wiki
git status
git add *.md
git commit -m "Update DomainPostTrain Wiki"
git push
```

Windows PowerShell:

```powershell
Copy-Item .wiki\*.md ..\REPO.wiki\
Set-Location ..\REPO.wiki
git status
git add *.md
git commit -m "Update DomainPostTrain Wiki"
git push
```

Expected result: after push, GitHub renders these pages in the repository Wiki.

## Permission warning

Check Wiki edit permissions before publishing. Public repository Wikis are public. Repository settings can restrict editing to users with write access, or can allow broader public editing depending on the repository configuration.

For a training pipeline repository, the conservative default is to restrict Wiki edits to trusted collaborators.

## Maintenance checklist

Before each release:

1. Update [Installation](Installation) for dependency changes.
2. Update [Configuration](Configuration) for new, renamed, or removed settings.
3. Update [Data Contracts](Data-Contracts) for row schema changes.
4. Update [GRPO And Reward Judge](GRPO-and-Reward-Judge) if reward behavior changes.
5. Move recurring support issues into [Troubleshooting](Troubleshooting) or [FAQ](FAQ).
6. Copy `.wiki/*.md` into the Wiki repository and push.

