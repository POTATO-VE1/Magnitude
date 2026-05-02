"""
Interactive image search CLI for VectorDB.

Loads the CLIP model ONCE at startup, then enters a REPL loop
so subsequent queries are near-instant (~100ms embed + ~5ms search).
Results are displayed as an HTML image grid in your browser.
"""

import os
import sys
import time
import tempfile
import webbrowser
import html

from embedder import CLIPEmbedder
from vectordb_client import VectorDBClient
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

console = Console()


def build_results_html(query: str, results: list, search_ms: float) -> str:
    """Build a self-contained HTML page showing the search results as an image grid."""
    cards = ""
    for i, res in enumerate(results):
        score = res.get("score", 0.0)
        distance = res.get("distance", 0.0)
        vid = res.get("id", 0)
        meta = res.get("metadata") or {}
        path = meta.get("path", "")
        filename = meta.get("filename", os.path.basename(path) if path else "unknown")

        if path and os.path.exists(path):
            # Use file:// URI so the browser can load local images
            img_src = f"file://{path}"
        else:
            img_src = ""

        score_pct = min(score * 100, 100)

        cards += f"""
        <div class="card" onclick="openModal('{img_src}', '{html.escape(filename)}', {score:.4f}, {vid})">
            <img class="card-img" src="{img_src}" alt="{html.escape(filename)}" loading="lazy"/>
            <div class="card-overlay">
                <span class="card-rank">{i + 1}</span>
            </div>
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
<title>VectorDB · {html.escape(query)}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    body {{
        font-family: 'Inter', system-ui, sans-serif;
        background: #fafafa;
        color: #1a1a1a;
        min-height: 100vh;
    }}

    /* ── Header ────────────────────────────────────────────── */
    .header {{
        background: #fff;
        border-bottom: 1px solid #e5e5e5;
        padding: 28px 40px;
    }}
    .header-inner {{
        max-width: 1200px;
        margin: 0 auto;
    }}
    .header-label {{
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #999;
        margin-bottom: 6px;
    }}
    .header-query {{
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a1a;
    }}
    .header-stats {{
        margin-top: 6px;
        font-size: 0.8rem;
        color: #888;
    }}
    .header-stats span {{
        display: inline-block;
        background: #f0f0f0;
        padding: 3px 10px;
        border-radius: 20px;
        margin-right: 6px;
    }}

    /* ── Grid ──────────────────────────────────────────────── */
    .grid {{
        max-width: 1200px;
        margin: 32px auto;
        padding: 0 40px;
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 20px;
    }}

    /* ── Card ──────────────────────────────────────────────── */
    .card {{
        background: #fff;
        border-radius: 10px;
        overflow: hidden;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border: 1px solid #ebebeb;
        position: relative;
    }}
    .card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }}

    .card-img {{
        width: 100%;
        height: 200px;
        object-fit: cover;
        display: block;
        background: #f2f2f2;
    }}

    .card-overlay {{
        position: absolute;
        top: 10px;
        left: 10px;
    }}
    .card-rank {{
        background: #fff;
        color: #1a1a1a;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.75rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.15);
    }}

    .card-body {{
        padding: 12px 14px 14px;
    }}
    .card-name {{
        font-size: 0.8rem;
        font-weight: 500;
        color: #333;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 8px;
    }}
    .card-bar-track {{
        height: 4px;
        background: #f0f0f0;
        border-radius: 2px;
        overflow: hidden;
        margin-bottom: 6px;
    }}
    .card-bar-fill {{
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #81C784);
        border-radius: 2px;
        transition: width 0.4s ease;
    }}
    .card-score {{
        font-size: 0.72rem;
        color: #999;
        font-weight: 500;
    }}

    /* ── Modal ─────────────────────────────────────────────── */
    .modal-bg {{
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.6);
        backdrop-filter: blur(4px);
        z-index: 100;
        justify-content: center;
        align-items: center;
    }}
    .modal-bg.active {{
        display: flex;
    }}
    .modal {{
        background: #fff;
        border-radius: 14px;
        max-width: 800px;
        max-height: 90vh;
        width: 90%;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,0.2);
        animation: modalIn 0.25s ease;
    }}
    @keyframes modalIn {{
        from {{ transform: scale(0.95); opacity: 0; }}
        to {{ transform: scale(1); opacity: 1; }}
    }}
    .modal img {{
        width: 100%;
        max-height: 70vh;
        object-fit: contain;
        background: #fafafa;
    }}
    .modal-info {{
        padding: 16px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid #eee;
    }}
    .modal-name {{
        font-weight: 600;
        font-size: 0.9rem;
        color: #333;
    }}
    .modal-meta {{
        font-size: 0.78rem;
        color: #999;
    }}
    .modal-close {{
        position: absolute;
        top: 16px;
        right: 16px;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: rgba(255,255,255,0.9);
        border: none;
        font-size: 1.1rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.15);
        z-index: 101;
    }}
    .modal-close:hover {{
        background: #fff;
    }}
</style>
</head>
<body>
    <div class="header">
        <div class="header-inner">
            <div class="header-label">VectorDB Image Search</div>
            <div class="header-query">{html.escape(query)}</div>
            <div class="header-stats">
                <span>{len(results)} results</span>
                <span>{search_ms:.0f} ms</span>
            </div>
        </div>
    </div>
    <div class="grid">
        {cards}
    </div>

    <!-- Lightbox modal -->
    <div class="modal-bg" id="modal" onclick="closeModal()">
        <div class="modal" onclick="event.stopPropagation()">
            <button class="modal-close" onclick="closeModal()">✕</button>
            <img id="modal-img" src="" alt=""/>
            <div class="modal-info">
                <span class="modal-name" id="modal-name"></span>
                <span class="modal-meta" id="modal-meta"></span>
            </div>
        </div>
    </div>

    <script>
    function openModal(src, name, score, vid) {{
        document.getElementById('modal-img').src = src;
        document.getElementById('modal-name').textContent = name;
        document.getElementById('modal-meta').textContent = 'Score: ' + score.toFixed(4) + '  ·  ID: ' + vid;
        document.getElementById('modal').classList.add('active');
    }}
    function closeModal() {{
        document.getElementById('modal').classList.remove('active');
    }}
    document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});
    </script>
</body>
</html>"""


def show_results(query: str, results: list, search_ms: float):
    """Display results both in the terminal table and as an HTML page in the browser."""
    # Terminal table
    table = Table(title=f"Results for '{query}' ({search_ms:.1f}ms)")
    table.add_column("Rank", justify="right", style="cyan", width=4)
    table.add_column("Score", justify="right", style="green", width=8)
    table.add_column("ID", justify="right", style="magenta", width=6)
    table.add_column("Filename", style="yellow")

    for i, res in enumerate(results):
        score = res.get("score", 0.0)
        vid = res.get("id", 0)
        meta = res.get("metadata") or {}
        filename = meta.get("filename", "N/A")
        table.add_row(str(i + 1), f"{score:.4f}", str(vid), filename)

    console.print(table)

    # HTML page
    page = build_results_html(query, results, search_ms)
    tmp = os.path.join(tempfile.gettempdir(), "vectordb_results.html")
    with open(tmp, "w") as f:
        f.write(page)
    webbrowser.open(f"file://{tmp}")
    console.print(f"[dim]Opened results in browser: {tmp}[/dim]\n")


def main():
    console.print("[bold cyan]━━━ VectorDB Image Search ━━━[/bold cyan]")
    console.print("[dim]Loading CLIP model (one-time)...[/dim]")

    t0 = time.time()
    embedder = CLIPEmbedder()
    client = VectorDBClient()
    load_time = time.time() - t0

    console.print(f"[green]✓ Model loaded in {load_time:.1f}s[/green]")

    # Resolve collection UUIDs once
    try:
        tenant_id = client.get_or_create_tenant("default")
        db_id = client.get_or_create_database(tenant_id, "images_db")
        col_id = client.get_or_create_collection(tenant_id, db_id, "clip_images", dimension=512)
    except Exception as e:
        console.print(f"[red]Failed to connect to VectorDB: {e}[/red]")
        return

    console.print("[green]✓ Connected to VectorDB[/green]")
    console.print("[dim]Type your search query below. Type 'quit' or 'exit' to stop.[/dim]\n")

    # Interactive REPL
    while True:
        try:
            query = Prompt.ask("[bold cyan]Search[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        query = query.strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        # ── Step 1: CLIP text embedding ──────────────────────────────
        t_embed_start = time.time()
        query_emb = embedder.embed_text(query)
        t_embed_end = time.time()
        embed_ms = (t_embed_end - t_embed_start) * 1000
        vec_dim = len(query_emb)

        # ── Step 2: HTTP request to VectorDB ─────────────────────────
        t_http_start = time.time()
        try:
            results = client.search_vectors(tenant_id, db_id, col_id, query_emb, top_k=5)
        except Exception as e:
            console.print(f"[red]Search failed: {e}[/red]\n")
            continue
        t_http_end = time.time()
        http_ms = (t_http_end - t_http_start) * 1000

        # ── Step 3: Parse results & metadata ─────────────────────────
        t_parse_start = time.time()
        n_results = len(results) if results else 0
        n_with_meta = sum(1 for r in (results or []) if r.get("metadata"))
        n_with_path = sum(1 for r in (results or []) if (r.get("metadata") or {}).get("path"))
        t_parse_end = time.time()
        parse_ms = (t_parse_end - t_parse_start) * 1000

        total_ms = embed_ms + http_ms + parse_ms

        # ── Print detailed breakdown ─────────────────────────────────
        console.print()
        console.print("[bold]┌─ Search Pipeline Breakdown ─────────────────────────┐[/bold]")
        console.print(f"│  [cyan]Query:[/cyan]           \"{query}\"")
        console.print(f"│  [cyan]Vector dim:[/cyan]      {vec_dim}")
        console.print(f"│")
        console.print(f"│  [yellow]① CLIP embed:[/yellow]    {embed_ms:>8.1f} ms   (text → {vec_dim}d float32)")
        console.print(f"│  [yellow]② HTTP search:[/yellow]   {http_ms:>8.1f} ms   (Go HNSW index lookup)")
        console.print(f"│  [yellow]③ Parse/meta:[/yellow]    {parse_ms:>8.1f} ms   ({n_results} results, {n_with_meta} with metadata)")
        console.print(f"│")
        console.print(f"│  [green bold]Total:[/green bold]           {total_ms:>8.1f} ms")
        console.print(f"│  [dim]Results:[/dim]         {n_results} returned, {n_with_path} have image paths")
        console.print("[bold]└────────────────────────────────────────────────────┘[/bold]")

        if not results:
            console.print("[yellow]No results found.[/yellow]\n")
            continue

        # ── Step 4: Render HTML & open browser ───────────────────────
        t_render_start = time.time()
        show_results(query, results, total_ms)
        t_render_end = time.time()
        render_ms = (t_render_end - t_render_start) * 1000
        console.print(f"[dim]  (HTML render + browser open: {render_ms:.0f}ms)[/dim]")


if __name__ == "__main__":
    main()
