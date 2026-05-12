"""Modern 3D Streamlit UI for neurons, synapses, and manual chat."""

from __future__ import annotations

import math
from functools import lru_cache

import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_plotly_events import plotly_events

from brain.config import Paths
from brain.router import Router
from brain.synapses import SynapseManager
from brain.vault import Vault
from brain.vectors import VectorStore


@st.cache_data(ttl=10)
def _load_nodes(limit_per_region: int = 200):
    nodes = []
    for region, path in Paths.REGIONS.items():
        if not path.exists():
            continue
        for f in sorted(path.glob("*.md"))[:limit_per_region]:
            nodes.append({"id": f.stem, "group": region, "path": str(f)})
    return nodes


@st.cache_data(ttl=10)
def _load_edges(limit: int = 1200):
    synapses = SynapseManager()
    return [{"from": s.source_id, "to": s.target_id, "value": int(s.strength)} for s in synapses.top_synapses(limit=limit)]


@lru_cache(maxsize=1)
def _region_colors():
    return {
        "prefrontal": "#7c3aed", "hippocampus": "#06b6d4", "creative": "#f59e0b",
        "predictive": "#10b981", "amygdala": "#ef4444", "executive": "#3b82f6",
    }


def _layout_positions(nodes: list[dict]) -> dict[str, tuple[float, float, float]]:
    by_region: dict[str, list[str]] = {}
    for n in nodes:
        by_region.setdefault(n["group"], []).append(n["id"])
    positions: dict[str, tuple[float, float, float]] = {}
    regions = list(by_region.keys())
    for r_idx, region in enumerate(regions):
        base = (2 * math.pi * r_idx) / max(1, len(regions))
        cx, cy = 5.2 * math.cos(base), 5.2 * math.sin(base)
        ids = by_region[region]
        for i, nid in enumerate(ids):
            a = (2 * math.pi * i) / max(1, len(ids))
            radius = 1.0 + (i % 12) * 0.14
            positions[nid] = (cx + radius * math.cos(a), cy + radius * math.sin(a), ((i % 13) - 6) * 0.2)
    return positions


def _build_3d_figure(nodes: list[dict], edges: list[dict], min_strength: int = 1, selected_node: str = ""):
    colors = _region_colors()
    pos = _layout_positions(nodes)
    ids = {n["id"] for n in nodes}

    ex, ey, ez = [], [], []
    for e in edges:
        if e["value"] < min_strength or e["from"] not in ids or e["to"] not in ids:
            continue
        x1, y1, z1 = pos[e["from"]]
        x2, y2, z2 = pos[e["to"]]
        ex += [x1, x2, None]; ey += [y1, y2, None]; ez += [z1, z2, None]

    edge_trace = go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color="rgba(148,163,184,0.28)", width=1), hoverinfo="none")

    nx, ny, nz, nt, nc, ns = [], [], [], [], [], []
    for n in nodes:
        x, y, z = pos[n["id"]]
        nx.append(x); ny.append(y); nz.append(z)
        nt.append(f"{n['id']}<br>region={n['group']}")
        nc.append(colors.get(n["group"], "#94a3b8"))
        ns.append(9 if n["id"] == selected_node else 5)

    node_trace = go.Scatter3d(
        x=nx, y=ny, z=nz, mode="markers", marker=dict(size=ns, color=nc, opacity=0.96),
        text=nt, customdata=[n["id"] for n in nodes], hovertemplate="%{text}<extra></extra>")

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        paper_bgcolor="#0b1020", plot_bgcolor="#0b1020", margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor="#0b1020", camera=dict(eye=dict(x=1.45, y=1.45, z=1.2))),
        showlegend=False,
    )
    return fig


def _neighbors(node_id: str, edges: list[dict]) -> list[tuple[str, int]]:
    out = []
    for e in edges:
        if e["from"] == node_id:
            out.append((e["to"], e["value"]))
        elif e["to"] == node_id:
            out.append((e["from"], e["value"]))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def _inject_css():
    st.markdown("""
    <style>
      .stApp {background: radial-gradient(1200px 850px at 12% -10%, #121d3d 0%, #0b1020 54%, #060912 100%); color:#e5e7eb;}
      .block-container {max-width:1400px; padding-top:1rem;}
      .stTabs [data-baseweb="tab"] {background:rgba(255,255,255,0.04); border-radius:12px;}
    </style>
    """, unsafe_allow_html=True)


def render():
    st.set_page_config(page_title="Digital Brain 3D", layout="wide")
    _inject_css()
    st_autorefresh(interval=4000, key="brain_refresh")  # realtime-ish update

    st.title("🧠 Digital Brain")
    st.caption("3D neural graph with zoom, click-to-inspect neurons, and live connection updates.")

    vault = Vault(); router = Router(); vectors = VectorStore()
    tab_graph, tab_chat, tab_stats = st.tabs(["3D Neural Graph", "Chat", "Live Stats"])

    with tab_graph:
        c1, c2 = st.columns([1, 1])
        node_cap = c1.slider("Nodes per region", 20, 300, 150, 10)
        edge_min = c2.slider("Min synapse strength", 1, 30, 2, 1)

        nodes = _load_nodes(node_cap)
        edges = _load_edges()
        node_ids = [n["id"] for n in nodes]

        selected = st.session_state.get("selected_neuron", "")
        fig = _build_3d_figure(nodes, edges, min_strength=edge_min, selected_node=selected)
        clicked = plotly_events(fig, click_event=True, select_event=False, hover_event=False, key="graph_click")
        if clicked:
            idx = clicked[0].get("pointIndex")
            if idx is not None and 0 <= idx < len(node_ids):
                st.session_state["selected_neuron"] = node_ids[idx]
                selected = node_ids[idx]

        st.caption("Use mouse wheel to zoom, drag to rotate, right-drag to pan.")

        manual_pick = st.selectbox("Or pick neuron", options=[""] + node_ids, index=(node_ids.index(selected)+1 if selected in node_ids else 0))
        if manual_pick:
            selected = manual_pick
            st.session_state["selected_neuron"] = selected

        if selected:
            st.subheader(f"Neuron Detail: {selected}")
            neuron = vault.read_neuron_by_id(selected)
            if neuron:
                st.write(f"**Title:** {neuron.title}")
                st.write(f"**Region:** {neuron.region}")
                st.write(f"**Type:** {neuron.type}")
            neigh = _neighbors(selected, edges)
            st.write("**Top Connections:**")
            st.dataframe([{"neighbor": n, "strength": s} for n, s in neigh[:25]], use_container_width=True)

    with tab_chat:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        prompt = st.chat_input("Ask manually. Use RECALL:/DECIDE:/CREATE: tags.")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    emb = router.get_embedding(prompt)
                    context = ""
                    if emb:
                        hits = vectors.hybrid_search(prompt, emb, top_k=4)
                        context = "\n\n".join(f"[{h['id']}] {h['document'][:260]}" for h in hits)
                    out = router.route(prompt, context=context)
                    st.markdown(out["response"])
                    st.caption(f"{out['mode']} • {out['provider']}")
            st.session_state.messages.append({"role": "assistant", "content": out["response"]})

    with tab_stats:
        counts = vault.count_neurons(); syn = SynapseManager().total_count(); vec = vectors.count()
        a,b,c,d = st.columns(4)
        a.metric("Neurons", counts.get("total", 0)); b.metric("Synapses", syn)
        c.metric("Vectors", sum(vec.values())); d.metric("Inbox", vault.inbox_count())
        st.json({"regions": counts, "vectors": vec})


def main():
    render()


if __name__ == "__main__":
    main()
