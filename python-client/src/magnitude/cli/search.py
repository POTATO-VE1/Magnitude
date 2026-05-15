"""Interactive image search CLI for Magnitude."""

import argparse
import base64
import json as json_module
import mimetypes
import os
import sys
import time
import tempfile
import webbrowser
import html


def _encode_image_data_uri(path: str) -> str:
    """Encode a local image file as a base64 data URI for inline embedding."""
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_results_html(query: str, results: list, search_ms: float) -> str:
    """Build a self-contained HTML page showing search results as an image grid."""
    cards = ""
    for i, res in enumerate(results):
        score = res.score
        vid = res.id
        meta = res.metadata or {}
        path = meta.get("path", "")
        filename = meta.get("filename", os.path.basename(path) if path else "unknown")

        # Use base64 data URIs so images load without a server (works in all browsers)
        if path and os.path.exists(path):
            try:
                img_src = _encode_image_data_uri(path)
            except (OSError, PermissionError):
                img_src = ""
        else:
            img_src = ""

        score_pct = min(score * 100, 100)

        # Safely escape for JavaScript context to prevent XSS.
        # json.dumps produces a JS-safe string literal; html.escape converts the
        # wrapping " to &quot; so it doesn't break the onclick attribute.
        img_src_js = html.escape(json_module.dumps(img_src))
        filename_js = html.escape(json_module.dumps(filename))

        cards += f"""
        <div class="card" onclick="openModal({img_src_js}, {filename_js}, {score:.4f}, {vid})">
            <img class="card-img" src="{html.escape(img_src, quote=True)}" alt="{html.escape(filename, quote=True)}" loading="lazy"/>
            <div class="card-overlay"><span class="card-rank">{i + 1}</span></div>
            <div class="card-body">
                <p class="card-name">{html.escape(filename)}</p>
                <div class="card-bar-track"><div class="card-bar-fill" style="width:{score_pct:.0f}%"></div></div>
                <p class="card-score">{score:.4f} match</p>
            </div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Magnitude · {html.escape(query)}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Inter', system-ui, sans-serif; background: #fafafa; color: #1a1a1a; min-height: 100vh; }}
    .header {{ background: #fff; border-bottom: 1px solid #e5e5e5; padding: 28px 40px; }}
    .header-inner {{ max-width: 1200px; margin: 0 auto; }}
    .header-label {{ font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #999; margin-bottom: 6px; }}
    .header-query {{ font-size: 1.6rem; font-weight: 700; }}
    .header-stats {{ margin-top: 6px; font-size: 0.8rem; color: #888; }}
    .header-stats span {{ display: inline-block; background: #f0f0f0; padding: 3px 10px; border-radius: 20px; margin-right: 6px; }}
    .grid {{ max-width: 1200px; margin: 32px auto; padding: 0 40px; display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 20px; }}
    .card {{ background: #fff; border-radius: 10px; overflow: hidden; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; border: 1px solid #ebebeb; position: relative; }}
    .card:hover {{ transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.08); }}
    .card-img {{ width: 100%; height: 200px; object-fit: cover; display: block; background: #f2f2f2; }}
    .card-overlay {{ position: absolute; top: 10px; left: 10px; }}
    .card-rank {{ background: #fff; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.75rem; box-shadow: 0 1px 4px rgba(0,0,0,0.15); }}
    .card-body {{ padding: 12px 14px 14px; }}
    .card-name {{ font-size: 0.8rem; font-weight: 500; color: #333; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 8px; }}
    .card-bar-track {{ height: 4px; background: #f0f0f0; border-radius: 2px; overflow: hidden; margin-bottom: 6px; }}
    .card-bar-fill {{ height: 100%; background: linear-gradient(90deg, #4CAF50, #81C784); border-radius: 2px; }}
    .card-score {{ font-size: 0.72rem; color: #999; font-weight: 500; }}
    .modal-bg {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.6); backdrop-filter: blur(4px); z-index: 100; justify-content: center; align-items: center; }}
    .modal-bg.active {{ display: flex; }}
    .modal {{ background: #fff; border-radius: 14px; max-width: 800px; max-height: 90vh; width: 90%; overflow: hidden; box-shadow: 0 20px 60px rgba(0,0,0,0.2); }}
    .modal img {{ width: 100%; max-height: 70vh; object-fit: contain; background: #fafafa; }}
    .modal-info {{ padding: 16px 20px; display: flex; justify-content: space-between; border-top: 1px solid #eee; }}
    .modal-name {{ font-weight: 600; font-size: 0.9rem; }}
    .modal-meta {{ font-size: 0.78rem; color: #999; }}
    .modal-close {{ position: absolute; top: 16px; right: 16px; width: 36px; height: 36px; border-radius: 50%; background: rgba(255,255,255,0.9); border: none; font-size: 1.1rem; cursor: pointer; box-shadow: 0 1px 4px rgba(0,0,0,0.15); z-index: 101; }}
</style>
</head>
<body>
    <div class="header"><div class="header-inner">
        <div class="header-label">Magnitude Image Search</div>
        <div class="header-query">{html.escape(query)}</div>
        <div class="header-stats"><span>{len(results)} results</span><span>{search_ms:.0f} ms</span></div>
    </div></div>
    <div class="grid">{cards}</div>
    <div class="modal-bg" id="modal" onclick="closeModal()">
        <div class="modal" onclick="event.stopPropagation()">
            <button class="modal-close" onclick="closeModal()">&#10005;</button>
            <img id="modal-img" src="" alt=""/>
            <div class="modal-info"><span class="modal-name" id="modal-name"></span><span class="modal-meta" id="modal-meta"></span></div>
        </div>
    </div>
    <script>
    function openModal(src, name, score, vid) {{
        document.getElementById('modal-img').src = src;
        document.getElementById('modal-name').textContent = name;
        document.getElementById('modal-meta').textContent = 'Score: ' + score.toFixed(4) + '  ·  ID: ' + vid;
        document.getElementById('modal').classList.add('active');
    }}
    function closeModal() {{ document.getElementById('modal').classList.remove('active'); }}
    document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});
    </script>
</body>
</html>"""


def show_results(query: str, results: list, search_ms: float) -> None:
    """Display results in terminal and open HTML page in browser."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(title=f"Results for '{query}' ({search_ms:.1f}ms)")
        table.add_column("Rank", justify="right", style="cyan", width=4)
        table.add_column("Score", justify="right", style="green", width=8)
        table.add_column("ID", justify="right", style="magenta", width=6)
        table.add_column("Filename", style="yellow")

        for i, res in enumerate(results):
            meta = res.metadata or {}
            table.add_row(
                str(i + 1), f"{res.score:.4f}", str(res.id), meta.get("filename", "N/A")
            )

        console.print(table)
    except ImportError:
        print(f"\nResults for '{query}' ({search_ms:.1f}ms)")
        for i, res in enumerate(results):
            meta = res.metadata or {}
            print(
                f"  {i + 1}. score={res.score:.4f}  id={res.id}  file={meta.get('filename', 'N/A')}"
            )

    page = build_results_html(query, results, search_ms)
    tmp = os.path.join(tempfile.gettempdir(), "magnitude_results.html")
    with open(tmp, "w") as f:
        f.write(page)
    webbrowser.open(f"file://{tmp}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive image search for Magnitude"
    )
    parser.add_argument(
        "--host", type=str, default="https://localhost:8443", help="Server URL"
    )
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument(
        "--collection", type=str, default="clip_images", help="Collection name"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    try:
        from magnitude import VectorDBClient, CLIPEmbedder
    except ImportError:
        print("Error: Install with pip install magnitude-client[all]")
        sys.exit(1)

    try:
        from rich.prompt import Prompt

        console_available = True
    except ImportError:
        console_available = False

    print("Loading CLIP model (one-time)...")
    t0 = time.time()
    embedder = CLIPEmbedder()
    client = VectorDBClient(args.host, api_key=args.api_key)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Verify collection exists
    try:
        col = client.get_collection(args.collection)
        print(f"Connected to collection: {col}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("Type your search query. Type 'quit' to exit.\n")

    while True:
        try:
            if console_available:
                query = Prompt.ask("[bold cyan]Search[/bold cyan]")
            else:
                query = input("Search: ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        query = query.strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Embed
        t1 = time.time()
        query_emb = embedder.embed_text(query)
        embed_ms = (time.time() - t1) * 1000

        # Search
        t2 = time.time()
        try:
            results = client.search(args.collection, query_emb, top_k=args.top_k)
        except Exception as e:
            print(f"Search failed: {e}")
            continue
        http_ms = (time.time() - t2) * 1000

        total_ms = embed_ms + http_ms

        print(
            f"  embed: {embed_ms:.0f}ms  search: {http_ms:.0f}ms  total: {total_ms:.0f}ms"
        )

        if not results:
            print("  No results found.\n")
            continue

        show_results(query, results, total_ms)
        print()


if __name__ == "__main__":
    main()
