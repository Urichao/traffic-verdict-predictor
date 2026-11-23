r"""免安裝 Git！透過 GitHub 官方 API 一鍵建立新分支並只推送有異動的程式碼檔案。

特點：
1. 支援開立新分支（--branch），以最新 main 為基底建立乾淨 Commit 與分支，保留所有組員的提交紀錄並可直接發起 Pull Request。
2. 自動比對 GitHub 目標分支內容，只上傳真正有修改或新增的檔案（PR diff 乾淨無雜訊）。
3. 自動排除 .env、PDF/PPTX 報告檔與暫存檔，並自動移除 web/model_bundle.json 中的雲端 API 金鑰，確保安全。

用法：
    C:\Users\urichiao\github.cmd --branch urichao/fullstack-upload-depreciation
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402, F401

SKIP_DIRS = {".venv", "__pycache__", ".pytest_cache", ".git", "raw", "interim", "results"}
SKIP_FILES = {
    ".env",
    "cases.parquet",
    "case_index.joblib",
    "weekly_report_slides.html",
    "team_tasks.html",
    "generate_weekly_ppt.py",
}
SKIP_SUFFIXES = {".pyc", ".pdf", ".pptx", ".docx"}


def should_include(rel_path: Path) -> bool:
    for part in rel_path.parts:
        if part in SKIP_DIRS:
            return False
    if rel_path.name in SKIP_FILES or rel_path.suffix.lower() in SKIP_SUFFIXES:
        return False
    return True


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("utf-8")
    return hashlib.sha1(header + data).hexdigest()


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


def _check_token(token: str) -> str | None:
    if not token:
        return None
    r = requests.get(
        "https://api.github.com/user",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=20,
    )
    if r.ok:
        return r.json()["login"]
    return None


def _save_token_to_env(token: str) -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        env_path.write_text(f"GITHUB_TOKEN={token}\n", encoding="utf-8")
        return
    lines = env_path.read_text(encoding="utf-8").splitlines()
    out = []
    replaced = False
    for ln in lines:
        if ln.startswith("GITHUB_TOKEN="):
            out.append(f"GITHUB_TOKEN={token}")
            replaced = True
        else:
            out.append(ln)
    if not replaced:
        out.append(f"GITHUB_TOKEN={token}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def push_to_github(
    token: str,
    repo_name: str,
    branch: str = "",
    message: str = "",
    only_paths: list[str] | None = None,
    private: bool = False,
) -> str:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    owner = _check_token(token)
    if not owner:
        raise RuntimeError("GitHub Token 驗證失敗或已過期，請重新產生具有 repo 權限的 Token。")

    print(f"[1/4] 正在確認 GitHub 儲存庫：{owner}/{repo_name} ...")
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
        r_create.raise_for_status()
        default_branch = r_create.json().get("default_branch", "main")
    else:
        r_repo.raise_for_status()
        default_branch = r_repo.json().get("default_branch", "main")

    target_branch = (branch or default_branch).strip()

    # 取得基底分支最新 Commit（若目標分支已存在則以目標分支為基底，否則從 default_branch 開出新分支）
    r_target_ref = requests.get(
        f"https://api.github.com/repos/{owner}/{repo_name}/git/ref/heads/{target_branch}",
        headers=headers,
        timeout=30,
    )
    branch_exists = r_target_ref.status_code == 200
    if branch_exists:
        base_commit_sha = r_target_ref.json()["object"]["sha"]
        print(f"  → 更新既有分支：{target_branch}（基底 {base_commit_sha[:7]}）")
    else:
        r_main_ref = requests.get(
            f"https://api.github.com/repos/{owner}/{repo_name}/git/ref/heads/{default_branch}",
            headers=headers,
            timeout=30,
        )
        r_main_ref.raise_for_status()
        base_commit_sha = r_main_ref.json()["object"]["sha"]
        print(f"  → 將從 {default_branch}（{base_commit_sha[:7]}）開立新分支：{target_branch}")

    r_commit = requests.get(
        f"https://api.github.com/repos/{owner}/{repo_name}/git/commits/{base_commit_sha}",
        headers=headers,
        timeout=30,
    )
    r_commit.raise_for_status()
    base_tree_sha = r_commit.json()["tree"]["sha"]

    # 讀取遠端 Tree 以比對檔案 SHA，只上傳真正有異動的檔案
    r_remote_tree = requests.get(
        f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/{base_tree_sha}?recursive=1",
        headers=headers,
        timeout=30,
    )
    r_remote_tree.raise_for_status()
    remote_shas = {
        item["path"]: item["sha"]
        for item in r_remote_tree.json().get("tree", [])
        if item.get("type") == "blob"
    }

    all_files = collect_files()
    if only_paths:
        allowed = {p.replace("\\", "/") for p in only_paths}
        all_files = [(rel, raw) for rel, raw in all_files if rel in allowed]

    changed_files = [
        (rel, raw) for rel, raw in all_files if remote_shas.get(rel) != git_blob_sha(raw)
    ]

    if not changed_files:
        print("所有檔案與遠端分支完全一致，沒有需要提交的異動。")
        return f"https://github.com/{owner}/{repo_name}/tree/{target_branch}"

    print(f"[2/4] 偵測到 {len(changed_files)} 個有異動的檔案，正在上傳至 GitHub ...")
    tree_entries = []
    for idx, (rel_posix, raw_bytes) in enumerate(changed_files, 1):
        status_tag = "新增" if rel_posix not in remote_shas else "修改"
        print(f"  - ({idx}/{len(changed_files)}) [{status_tag}] {rel_posix} ({len(raw_bytes)/1024:.1f} KB)")
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

    print("[3/4] 正在建立 Git Commit ...")
    r_tree = requests.post(
        f"https://api.github.com/repos/{owner}/{repo_name}/git/trees",
        headers=headers,
        json={"base_tree": base_tree_sha, "tree": tree_entries},
        timeout=60,
    )
    r_tree.raise_for_status()
    new_tree_sha = r_tree.json()["sha"]

    commit_msg = (
        message.strip()
        or "新增文字檔上傳、車輛零件折舊試算、擴充表單欄位並放寬約字月收入擷取"
    )
    r_new_commit = requests.post(
        f"https://api.github.com/repos/{owner}/{repo_name}/git/commits",
        headers=headers,
        json={
            "message": commit_msg,
            "tree": new_tree_sha,
            "parents": [base_commit_sha],
        },
        timeout=30,
    )
    r_new_commit.raise_for_status()
    new_commit_sha = r_new_commit.json()["sha"]

    print(f"[4/4] 正在推送分支 refs/heads/{target_branch} ...")
    if branch_exists:
        r_ref_op = requests.patch(
            f"https://api.github.com/repos/{owner}/{repo_name}/git/refs/heads/{target_branch}",
            headers=headers,
            json={"sha": new_commit_sha, "force": False},
            timeout=30,
        )
    else:
        r_ref_op = requests.post(
            f"https://api.github.com/repos/{owner}/{repo_name}/git/refs",
            headers=headers,
            json={"ref": f"refs/heads/{target_branch}", "sha": new_commit_sha},
            timeout=30,
        )
    r_ref_op.raise_for_status()

    branch_url = f"https://github.com/{owner}/{repo_name}/tree/{target_branch}"
    pr_url = f"https://github.com/{owner}/{repo_name}/compare/{default_branch}...{target_branch}?expand=1"
    print("\n============================================================")
    print(f"🎉 GitHub 分支推送成功！（Commit: {new_commit_sha[:7]}）")
    print(f"👉 分支網址：{branch_url}")
    if target_branch != default_branch:
        print(f"👉 一鍵發起 Pull Request：{pr_url}")
    print("============================================================")
    return branch_url


def main() -> int:
    ap = argparse.ArgumentParser(description="一鍵建立 GitHub 分支並只推送有異動的檔案（免安裝 git.exe）")
    ap.add_argument("--token", default=os.getenv("GITHUB_TOKEN", ""), help="GitHub Personal Access Token (ghp_... / github_pat_...)")
    ap.add_argument("--repo", default="traffic-verdict-predictor", help="GitHub 儲存庫名稱（預設：traffic-verdict-predictor）")
    ap.add_argument("--branch", default="urichao/fullstack-upload-depreciation", help="目標分支名稱（預設：urichao/fullstack-upload-depreciation）")
    ap.add_argument("--message", "-m", default="", help="Commit 說明訊息")
    ap.add_argument("--only", nargs="*", default=None, help="僅比對並推送指定檔案路徑（省略則自動比對全專案）")
    ap.add_argument("--private", action="store_true", help="新建儲存庫時是否設為私人")
    args, unknown = ap.parse_known_args()

    token = (args.token or "").strip()
    for item in unknown:
        s = item.strip()
        if s.startswith(("ghp_", "github_pat_")):
            token = s

    if not _check_token(token):
        if token:
            print("偵測到本機 .env 中舊的 GitHub Token 已過期或無效，請貼上新的 Token。")
        token = input("請貼上有效的 GitHub Personal Access Token (ghp_... 或 github_pat_...)：").strip()

    if not token or not _check_token(token):
        print("GitHub Token 驗證失敗，已取消。")
        return 1

    _save_token_to_env(token)
    push_to_github(
        token=token,
        repo_name=args.repo,
        branch=args.branch,
        message=args.message,
        only_paths=args.only,
        private=args.private,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
