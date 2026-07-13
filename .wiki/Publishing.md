# 发布 Wiki / Publishing

## 中文

当你要把主仓库 `.wiki/` 源文件发布到 GitHub Wiki 仓库时，使用本页。

GitHub Wiki 是独立 Git 仓库。只在主仓库创建 `.wiki/` 不会让页面自动上线。

## GitHub Wiki 机制

根据 GitHub Docs，GitHub Wiki 的发布规则包括：

- Wiki 页面可以在 GitHub 网页上编辑，也可以本地编辑。
- Wiki 至少有一个初始页面后才能 clone。
- clone URL 形如 `https://github.com/OWNER/REPO.wiki.git`。
- 只有推送到 Wiki 默认分支的变更才会对读者生效。
- 文件名决定页面标题。
- 文件扩展名决定渲染方式。
- `_Sidebar.md` 会作为自定义侧边栏。
- 避免这些文件名字符：`\ / : * ? " < > |`。

官方参考：

- https://docs.github.com/en/communities/documenting-your-project-with-wikis/adding-or-editing-wiki-pages
- https://docs.github.com/en/communities/documenting-your-project-with-wikis/creating-a-footer-or-sidebar-for-your-wiki
- https://docs.github.com/en/communities/documenting-your-project-with-wikis/changing-access-permissions-for-wikis

## 一次性设置

1. 在 GitHub 打开仓库。
2. 打开 Wiki tab。
3. 如果 Wiki 为空，先创建一个初始页面。
4. clone Wiki 仓库：

```bash
git clone https://github.com/OWNER/REPO.wiki.git
```

把 `OWNER` 和 `REPO` 替换成真实仓库 owner 和名称。

## 从 `.wiki` 发布

在主仓库根目录执行：

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

预期结果：push 后，GitHub 会在仓库 Wiki 中渲染这些页面。

## 权限提醒

发布前检查 Wiki 编辑权限。公开仓库的 Wiki 是公开可读的；仓库设置可以限制只有写权限用户能编辑，也可能允许更广泛的 GitHub 用户编辑，取决于配置。

对训练流水线仓库，保守默认值是只允许可信协作者编辑 Wiki。

## 双语维护规则

本 Wiki 默认中文优先。维护时请保持：

- 中文内容在每页上半部分。
- 英文内容在 `---` 分隔线之后。
- 文件名继续使用 ASCII，避免跨平台同步和 Wiki URL 问题。
- `_Sidebar.md` 使用中文导航为默认，英文作为辅助标题。

## 发布前维护清单

每次发布前：

1. 依赖变化时更新 [安装](Installation)。
2. 新增、重命名或删除配置项时更新 [配置](Configuration)。
3. 行 schema 变化时更新 [数据契约](Data-Contracts)。
4. 奖励行为变化时更新 [GRPO 与 Reward Judge](GRPO-and-Reward-Judge)。
5. 把重复出现的支持问题整理进 [故障排查](Troubleshooting) 或 [常见问题](FAQ)。
6. 把 `.wiki/*.md` 复制到 Wiki 仓库并 push。

---

## English

Use this page when publishing `.wiki/` source files to the GitHub Wiki repository.

GitHub Wikis are Git repositories. Creating `.wiki/` in this main repository does not automatically make pages live on GitHub.

## GitHub Wiki Mechanics

According to GitHub Docs, GitHub Wiki publishing works as follows:

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

## One-Time Setup

1. In GitHub, open the repository.
2. Open the Wiki tab.
3. Create an initial page if the Wiki is empty.
4. Clone the Wiki repository:

```bash
git clone https://github.com/OWNER/REPO.wiki.git
```

Replace `OWNER` and `REPO` with the real repository owner and name.

## Publish from This Source Directory

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

## Permission Warning

Check Wiki edit permissions before publishing. Public repository Wikis are public. Repository settings can restrict editing to users with write access, or can allow broader public editing depending on the repository configuration.

For a training pipeline repository, the conservative default is to restrict Wiki edits to trusted collaborators.

## Bilingual Maintenance Rules

This Wiki is Chinese-first by default. When maintaining pages:

- Keep Chinese content in the upper half of each page.
- Put English content after the `---` separator.
- Keep filenames ASCII to avoid cross-platform sync and Wiki URL issues.
- Keep `_Sidebar.md` Chinese-first, with English labels as secondary cues.

## Maintenance Checklist

Before each release:

1. Update [Installation](Installation) for dependency changes.
2. Update [Configuration](Configuration) for new, renamed, or removed settings.
3. Update [Data Contracts](Data-Contracts) for row schema changes.
4. Update [GRPO And Reward Judge](GRPO-and-Reward-Judge) if reward behavior changes.
5. Move recurring support issues into [Troubleshooting](Troubleshooting) or [FAQ](FAQ).
6. Copy `.wiki/*.md` into the Wiki repository and push.
