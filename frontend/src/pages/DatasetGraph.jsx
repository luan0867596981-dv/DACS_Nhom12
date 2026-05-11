import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Search, Download, Maximize, PlayCircle, Network } from 'lucide-react';
import ForceGraph2D from 'react-force-graph-2d';

export default function DatasetGraph() {
  const [dataset, setDataset] = useState('C');
  const [drugLimit, setDrugLimit] = useState(30);
  const [diseaseLimit, setDiseaseLimit] = useState(60);
  const [showProtein, setShowProtein] = useState(false);
  const [search, setSearch] = useState('');
  
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showLabels, setShowLabels] = useState(false);
  
  const [hoverNode, setHoverNode] = useState(null);
  const [highlightNodes, setHighlightNodes] = useState(new Set());
  const [highlightLinks, setHighlightLinks] = useState(new Set());
  
  const fgRef = useRef();

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const r = await fetch(`http://127.0.0.1:8000/graph/network?dataset=${dataset}&drug_limit=${drugLimit}&disease_limit=${diseaseLimit}&show_protein=${showProtein}&search=${encodeURIComponent(search)}`);
      const d = await r.json();
      setGraphData({ nodes: d.nodes, links: d.edges });
      setStats(d.stats);
      if (fgRef.current) {
        setTimeout(() => fgRef.current.zoomToFit(400, 50), 500);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleFit = () => {
    if (fgRef.current) fgRef.current.zoomToFit(400, 50);
  };

  const handleExport = () => {
    if (!fgRef.current) return;
    const canvas = document.querySelector('.force-graph-container canvas');
    if (canvas) {
      const link = document.createElement('a');
      link.download = `AMNTDDA_Graph_${dataset}.png`;
      link.href = canvas.toDataURL();
      link.click();
    }
  };

  const handleNodeHover = useCallback((n) => {
    const nH = new Set();
    const lH = new Set();
    if (n) {
      nH.add(n.id);
      graphData.links.forEach(l => {
        const s = typeof l.source === 'object' ? l.source.id : l.source;
        const t = typeof l.target === 'object' ? l.target.id : l.target;
        if (s === n.id || t === n.id) {
          lH.add(l);
          nH.add(s);
          nH.add(t);
        }
      });
    }
    setHoverNode(n);
    setHighlightNodes(nH);
    setHighlightLinks(lH);
  }, [graphData]);

  const paintNode = useCallback((n, ctx, scale) => {
    const isHi = highlightNodes.has(n.id);
    const isDim = hoverNode && !isHi;
    
    ctx.save();
    ctx.globalAlpha = isDim ? 0.15 : 1;
    
    let radius = 5;
    let color = '#10b981'; // protein
    
    if (n.group === 'drug') { radius = 8; color = '#3b82f6'; }
    else if (n.group === 'disease') { radius = 6; color = '#ef4444'; }
    
    ctx.beginPath();
    ctx.arc(n.x, n.y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    
    if (isHi) {
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2 / scale;
      ctx.stroke();
    }
    
    if (showLabels || isHi || scale > 3) {
      const fs = 10 / scale;
      ctx.font = `bold ${fs}px Sans-Serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#e2e8f0'; // dark theme text
      ctx.fillText(n.label, n.x, n.y + radius + 2/scale + fs/2);
    }
    
    ctx.restore();
  }, [hoverNode, highlightNodes, showLabels]);

  const paintLink = useCallback((l, ctx, scale) => {
    const isHi = highlightLinks.has(l);
    ctx.save();
    ctx.globalAlpha = hoverNode && !isHi ? 0.1 : 0.4;
    ctx.strokeStyle = isHi ? '#f59e0b' : '#94a3b8';
    ctx.lineWidth = isHi ? 2 / scale : 1 / scale;
    ctx.beginPath();
    ctx.moveTo(l.source.x, l.source.y);
    ctx.lineTo(l.target.x, l.target.y);
    ctx.stroke();
    ctx.restore();
  }, [hoverNode, highlightLinks]);

  return (
    <div className="flex-1 flex h-full overflow-hidden bg-slate-50 dark:bg-slate-950 font-sans">
      
      {/* LEFT COL: Controls */}
      <div className="w-full md:w-[380px] flex-shrink-0 flex flex-col bg-white dark:bg-slate-900 border-r dark:border-slate-800 shadow-2xl z-30 p-8 space-y-8 overflow-y-auto">
        <div>
          <h2 className="text-2xl font-black text-slate-800 dark:text-slate-100 flex items-center gap-3"><Network className="text-teal-500"/> Biểu đồ liên kết</h2>
          <p className="text-xs font-bold text-slate-400 mt-2">Trực quan hoá mạng lưới Thuốc–Bệnh–Protein</p>
        </div>

        <div className="space-y-6">
          <div className="space-y-3">
            <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Dataset</label>
            <select value={dataset} onChange={e=>setDataset(e.target.value)} className="w-full p-4 bg-slate-50 dark:bg-slate-800 rounded-2xl border dark:border-slate-700 font-bold outline-none text-sm focus:ring-2 ring-teal-500/50">
              <option value="C">C-dataset (663 thuốc, 409 bệnh)</option>
              <option value="B">B-dataset (269 thuốc, 598 bệnh)</option>
              <option value="F">F-dataset (593 thuốc, 313 bệnh)</option>
            </select>
          </div>

          <div className="space-y-4">
            <div className="flex justify-between"><label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Số thuốc hiển thị</label><span className="font-black text-teal-600">{drugLimit}</span></div>
            <input type="range" min="10" max="200" step="5" value={drugLimit} onChange={e=>setDrugLimit(parseInt(e.target.value))} className="w-full accent-teal-600"/>
          </div>

          <div className="space-y-4">
            <div className="flex justify-between"><label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Số bệnh hiển thị tối đa</label><span className="font-black text-rose-600">{diseaseLimit}</span></div>
            <input type="range" min="10" max="300" step="5" value={diseaseLimit} onChange={e=>setDiseaseLimit(parseInt(e.target.value))} className="w-full accent-rose-600"/>
          </div>

          <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-800/50 rounded-2xl">
            <label className="text-xs font-black uppercase text-slate-600 dark:text-slate-300">Hiện Protein</label>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" checked={showProtein} onChange={e=>setShowProtein(e.target.checked)} />
              <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer dark:bg-slate-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-slate-600 peer-checked:bg-violet-500"></div>
            </label>
          </div>

          <div className="space-y-3">
            <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Tìm nhanh</label>
            <div className="relative">
              <Search size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" />
              <input type="text" placeholder="Tìm kiếm thuốc / bệnh..." value={search} onChange={e=>setSearch(e.target.value)} className="w-full pl-12 pr-4 py-4 bg-slate-50 dark:bg-slate-800 rounded-2xl border dark:border-slate-700 text-sm font-bold outline-none focus:ring-2 ring-teal-500/50" />
            </div>
          </div>

          <button onClick={handleGenerate} disabled={loading} className="w-full py-4 bg-teal-600 hover:bg-teal-500 text-white rounded-2xl font-black shadow-lg shadow-teal-900/20 flex items-center justify-center gap-2 transition-all active:scale-95">
            {loading ? <span className="animate-pulse">ĐANG TẢI...</span> : <><PlayCircle size={18}/> TẠO BIỂU ĐỒ</>}
          </button>
        </div>

        {/* Legend */}
        <div className="pt-6 border-t dark:border-slate-800 space-y-3">
          <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Chú thích</h4>
          <div className="grid grid-cols-2 gap-3 text-xs font-bold text-slate-600 dark:text-slate-300">
            <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-blue-500"></div> Thuốc</div>
            <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-red-500"></div> Bệnh</div>
            <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-emerald-500"></div> Protein</div>
            <div className="flex items-center gap-2"><div className="w-4 h-0.5 bg-slate-400"></div> Liên kết</div>
          </div>
        </div>
      </div>

      {/* RIGHT COL: Graph */}
      <div className="flex-1 relative flex flex-col bg-slate-900 force-graph-container">
        
        {/* Header Overlay */}
        <div className="absolute top-0 left-0 right-0 p-6 flex justify-between items-start z-10 pointer-events-none">
          <div className="pointer-events-auto bg-slate-900/50 backdrop-blur-md border border-white/10 px-6 py-4 rounded-3xl">
            <h3 className="font-black text-white text-lg">Mạng lưới liên kết {dataset}-dataset</h3>
            {stats && (
              <p className="text-xs font-bold text-teal-300 mt-1">
                {stats.drug_count} thuốc · {stats.disease_count} bệnh · {stats.total_edges} liên kết
              </p>
            )}
          </div>
          
          <div className="pointer-events-auto flex items-center gap-2 bg-slate-900/50 backdrop-blur-md border border-white/10 p-2 rounded-2xl">
            <button onClick={handleFit} title="Zoom to Fit" className="p-2.5 text-white/70 hover:text-white hover:bg-white/10 rounded-xl transition-all"><Maximize size={18}/></button>
            <button onClick={handleExport} title="Export PNG" className="p-2.5 text-white/70 hover:text-white hover:bg-white/10 rounded-xl transition-all"><Download size={18}/></button>
            <button onClick={()=>setShowLabels(!showLabels)} title="Toggle Labels" className={`p-2.5 rounded-xl font-black text-xs transition-all ${showLabels ? 'bg-teal-500 text-white' : 'text-white/70 hover:text-white hover:bg-white/10'}`}>TEXT</button>
          </div>
        </div>

        {/* Canvas */}
        <div className="flex-1 w-full h-full relative">
          {graphData.nodes.length === 0 && !loading ? (
            <div className="absolute inset-0 flex items-center justify-center text-slate-500 font-bold uppercase tracking-widest text-sm">
              Nhấn 'Tạo biểu đồ' để bắt đầu
            </div>
          ) : (
            <ForceGraph2D
              ref={fgRef}
              graphData={graphData}
              backgroundColor="#0f172a"
              nodeCanvasObject={paintNode}
              linkCanvasObject={paintLink}
              onNodeHover={handleNodeHover}
              onNodeClick={n => handleNodeHover(n)}
              cooldownTicks={100}
            />
          )}
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm z-20">
              <div className="w-12 h-12 border-4 border-teal-500 border-t-transparent rounded-full animate-spin"></div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
