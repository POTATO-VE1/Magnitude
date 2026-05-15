const searchInput = document.getElementById('search-input');
const resultsGrid = document.getElementById('results-grid');
const latencyInfo = document.getElementById('latency-info');
const embedTimeEl = document.getElementById('embed-time');
const searchTimeEl = document.getElementById('search-time');
const resCountEl = document.getElementById('res-count');
const collectionSelect = document.getElementById('collection-select');

let debounceTimer;

// Fetch collections on load
async function loadCollections() {
    try {
        const response = await fetch('/api/collections');
        if (!response.ok) throw new Error('Failed to load collections');
        
        const data = await response.json();
        const collections = data.collections || [];
        
        collectionSelect.innerHTML = '';
        if (collections.length === 0) {
            const opt = document.createElement('option');
            opt.value = "";
            opt.textContent = "No datasets found";
            collectionSelect.appendChild(opt);
            return;
        }

        collections.forEach(c => {
            const opt = document.createElement('option');
            opt.value = c.id;
            opt.textContent = c.name + ` (${c.vector_count || 0} imgs)`;
            // Select clip_images by default if it exists
            if (c.name === 'clip_images') opt.selected = true;
            collectionSelect.appendChild(opt);
        });
    } catch (err) {
        console.error(err);
        collectionSelect.innerHTML = '<option value="">Error loading datasets</option>';
    }
}
loadCollections();

// Auto-focus logic to keep the user engaged
document.addEventListener('keydown', () => {
    if (document.activeElement !== searchInput) {
        searchInput.focus();
    }
});

searchInput.addEventListener('input', (e) => {
    clearTimeout(debounceTimer);
    const query = e.target.value.trim();
    
    if (!query) {
        resultsGrid.innerHTML = '';
        latencyInfo.style.display = 'none';
        resCountEl.textContent = '0';
        return;
    }

    // Debounce to prevent spamming the backend
    debounceTimer = setTimeout(() => {
        performSearch(query);
    }, 300);
});

async function performSearch(query) {
    try {
        const collection_id = collectionSelect.value;
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, collection_id })
        });
        
        if (!response.ok) throw new Error('Search failed');
        
        const data = await response.json();
        renderResults(data.results);
        updateMetrics(data.metrics, data.results.length);
    } catch (err) {
        console.error(err);
    }
}

function updateMetrics(metrics, count) {
    latencyInfo.style.display = 'flex';
    embedTimeEl.textContent = metrics.embed_ms.toFixed(0);
    searchTimeEl.textContent = metrics.search_ms.toFixed(0);
    resCountEl.textContent = count;
}

function renderResults(results) {
    resultsGrid.innerHTML = '';
    
    if (results.length === 0) {
        resultsGrid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: #444; margin-top: 40px;">No results found.</div>';
        return;
    }

    results.forEach((res, index) => {
        const scorePct = Math.min(res.score * 100, 100);
        // Serve local image securely via the python backend proxy
        const imgSrc = `/api/image?path=${encodeURIComponent(res.path)}`;
        
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
            <div class="card-img-wrapper">
                <div class="card-rank">[${index + 1}]</div>
                <img src="${imgSrc}" class="card-img" alt="${res.filename}" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9IiMxMTEiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxNHB4IiBmaWxsPSIjNDQ0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjM1ZW0iPkltYWdlIE5vdCBGb3VuZDwvdGV4dD48L3N2Zz4='">
            </div>
            <div class="card-body">
                <div class="card-title">${res.filename}</div>
                <div class="card-match-label">
                    <span>MATCH</span>
                    <span style="color: #ff5722">${scorePct.toFixed(1)}%</span>
                </div>
                <div class="card-bar-bg">
                    <div class="card-bar-fill" style="width: ${scorePct}%"></div>
                </div>
            </div>
        `;
        resultsGrid.appendChild(card);
    });
}
