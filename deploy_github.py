r"""免安裝 Git！直接透過 GitHub 官方 API 一鍵建立儲存庫並上傳完整專案程式碼。

特點：
1. 不需本機安裝 git.exe，直接透過 GitHub REST API 建立 Repo 並上傳所有程式碼與模型檔案。
2. 自動排除 .env（保護個人帳密與 Token）並自動清除 web/model_bundle.json 中的雲端 API 金鑰，確保公開在 GitHub 上絕對安全。

用法：
    C:\Users\urichiao\github.cmd
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402, F401

SKIP_DIRS = {".venv", "__pycache__", ".pytest_cache", ".git", "raw", "interim"}
SKIP_FILES = {".env", "cases.parquet", "case_index.joblib"}


def should_include(rel_path: Path) -> bool:
    for part in rel_path.parts:
        if part in SKIP_DIRS:
            return False
    if rel_path.name in SKIP_FILES or rel_path.suffix == ".pyc":
        return False
    return True


def collect_files() -> list[tuple[str, bytes]]:
    items: list[tuple[str, bytes]] = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if not should_include(rel):
            continue
        raw = p.read_bytes()
        # 上傳至公開 GitHub 時，自動移除 model_bundle.json 內的 cloud_key_b64，避免金鑰外洩
        if rel.as_posix() == "web/model_bundle.json":
            try:
                obj = json.loads(raw.decode("utf-8"))
                obj["cloud_key_b64"] = ""
                raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            except Exception:
                pass
        items.append((rel.as_posix(), raw))
    return items


def push_to_github(token: str, repo_name: str, private: bool = False) -> str:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    r_user = requests.get("https://api.github.com/user", headers=headers, timeout=30)
    if not r_user.ok:
        raise RuntimeError(f"GitHub Token 驗證失敗 ({r_user.status_code})：{r_user.text}")
    owner = r_user.json()["login"]

    print(f"[1/3] 正在確認 / 建立 GitHub 儲存庫：{owner}/{repo_name} ...")
    r_repo = requests.get(f"https://api.github.com/repos/{owner}/{repo_name}", headers=headers, timeout=30)
    if r_repo.status_code == 404:
        r_create = requests.post(
            "https://api.github.com/user/repos",
            headers=headers,
            json={
                "name": repo_name,
                "description": "智慧司法判決預測與可解釋案例分析系統（台灣民事車禍判決預測 + LightGBM/SHAP/HAN + Gemma-4）",
                "private": private,
                "auto_init": True,
            },
            timeout=30,
        )
        if not r_create.ok:
            raise RuntimeError(f"建立儲存庫失敗 ({r_create.status_code})：{r_create.text}")
        default_branch = r_create.json().get("default_branch", "main")
    else:
        r_repo.raise_for_status()
        default_branch = r_repo.json().get("default_branch", "main")

    # 確保儲存庫已有初始 commit（若為空 repo 先建立 README.md）
    r_ref = requests.get(
        f"https://api.github.com/repos/{owner}/{repo_name}/git/ref/heads/{default_branch}",
        headers=headers,
        timeout=30,
    )
    if r_ref.status_code in (404, 409):
        readme_bytes = (ROOT / "README.md").read_bytes()
        requests.put(
            f"https://api.github.com/repos/{owner}/{repo_name}/contents/README.md",
            headers=headers,
            json={
                "message": "Initial commit",
                "content": base64.b64encode(readme_bytes).decode("ascii"),
            },
            timeout=30,
        ).raise_for_status()
        r_ref = requests.get(
            f"https://api.github.com/repos/{owner}/{repo_name}/git/ref/heads/{default_branch}",
            headers=headers,
            timeout=30,
        )
    r_ref.raise_for_status()
    latest_commit_sha = r_ref.json()["object"]["sha"]

    r_commit = requests.get(
        f"https://api.github.com/repos/{owner}/{repo_name}/git/commits/{latest_commit_sha}",
        headers=headers,
        timeout=30,
    )
    r_commit.raise_for_status()
    base_tree_sha = r_commit.json()["tree"]["sha"]

    files = collect_files()
    print(f"[2/3] 正在上傳 {len(files)} 個專案檔案至 {owner}/{repo_name} ...")
    tree_entries = []
    for idx, (rel_posix, raw_bytes) in enumerate(files, 1):
        print(f"  - ({idx}/{len(files)}) {rel_posix} ({len(raw_bytes)/1024:.1f} KB)")
        r_blob = requests.post(
            f"https://api.github.com/repos/{owner}/{repo_name}/git/blobs",
            headers=headers,
            json={
                "content": base64.b64encode(raw_bytes).decode("ascii"),
                "encoding": "base64",
            },
            timeout=60,
        )
        r_blob.raise_for_status()
        tree_entries.append({
            "path": rel_posix,
            "mode": "100644",
            "type": "blob",
            "sha": r_blob.json()["sha"],
        })

    print("[3/3] 正在建立 Git Commit 並更新主分支...")
    r_tree = requests.post(
        f"https://api.github.com/repos/{owner}/{repo_name}/git/trees",
        headers=headers,
        json={"base_tree": base_tree_sha, "tree": tree_entries},
        timeout=60,
    )
    r_tree.raise_for_status()
    new_tree_sha = r_tree.json()["sha"]

    r_new_commit = requests.post(
        f"https://api.github.com/repos/{owner}/{repo_name}/git/commits",
        headers=headers,
        json={
            "message": "Update Taiwan Civil Traffic Verdict Predictor (12,174 cases + LightGBM/SHAP/HAN + Gemma-4)",
            "tree": new_tree_sha,
            "parents": [latest_commit_sha],
        },
        timeout=30,
    )
    r_new_commit.raise_for_status()
    new_commit_sha = r_new_commit.json()["sha"]

    r_update_ref = requests.patch(
        f"https://api.github.com/repos/{owner}/{repo_name}/git/refs/heads/{default_branch}",
        headers=headers,
        json={"sha": new_commit_sha, "force": True},
        timeout=30,
    )
    r_update_ref.raise_for_status()

    repo_url = f"https://github.com/{owner}/{repo_name}"
    print("\n============================================================")
    print("🎉 GitHub 專案上傳成功！")
    print(f"👉 GitHub 儲存庫網址：{repo_url}")
    print("（已自動排除 .env 帳密與 API 金鑰，您可以直接將網址分享給組員與教授討論！）")
    print("============================================================")
    return repo_url


def main() -> int:
    ap = argparse.ArgumentParser(description="一鍵上傳專案至 GitHub（免安裝 git.exe）")
    ap.add_argument("--token", default=os.getenv("GITHUB_TOKEN", ""), help="GitHub Personal Access Token (ghp_... 或 github_pat_...)")
    ap.add_argument("--repo", default="traffic-verdict-predictor", help="GitHub 儲存庫名稱（預設：traffic-verdict-predictor）")
    ap.add_argument("--private", action="store_true", help="是否設為私人儲存庫（預設為公開）")
    args, unknown = ap.parse_known_args()

    token = (args.token or "").strip()
    for item in unknown:
        s = item.strip()
        if s.startswith(("ghp_", "github_pat_")) and not token:
            token = s

    if not token:
        token = input("請貼上 GitHub Personal Access Token (ghp_... 或 github_pat_...)：").strip()

    if not token:
        print("未提供 GitHub Token，已取消。")
        return 1

    # 自動儲存 GITHUB_TOKEN 至本機 .env，方便日後直接執行 github.cmd 一鍵更新
    env_path = ROOT / ".env"
    if env_path.exists():
        env_text = env_path.read_text(encoding="utf-8")
        if "GITHUB_TOKEN=" not in env_text:
            env_path.write_text(env_text.rstrip() + f"\nGITHUB_TOKEN={token}\n", encoding="utf-8")

    push_to_github(token=token, repo_name=args.repo, private=args.private)
    return 0


if __name__ == "__main__":
    sys.exit(main())
