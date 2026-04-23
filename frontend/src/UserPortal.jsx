import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Search, Moon, Sun, User, FileOutput, History, AlertCircle, PlayCircle, Network, ArrowRight, Table as TableIcon, Home as HomeIcon, CheckSquare, BarChart2 
} from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import ForceGraph2D from 'react-force-graph-2d';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

// Performance Mock for BarChart (3 Datasets)
const performanceDataB = [
  { metric: 'AUC', AMDGT: 0.932, OurImproved: 0.965 },
  { metric: 'AUPR', AMDGT: 0.915, OurImproved: 0.952 },
  { metric: 'F1', AMDGT: 0.860, OurImproved: 0.901 },
  { metric: 'MCC', AMDGT: 0.810, OurImproved: 0.870 },
];
const performanceDataC = [
  { metric: 'AUC', AMDGT: 0.825, OurImproved: 0.875 },
  { metric: 'AUPR', AMDGT: 0.812, OurImproved: 0.854 },
  { metric: 'F1', AMDGT: 0.760, OurImproved: 0.801 },
  { metric: 'MCC', AMDGT: 0.680, OurImproved: 0.720 },
];
const performanceDataF = [
  { metric: 'AUC', AMDGT: 0.880, OurImproved: 0.920 },
  { metric: 'AUPR', AMDGT: 0.850, OurImproved: 0.895 },
  { metric: 'F1', AMDGT: 0.820, OurImproved: 0.865 },
  { metric: 'MCC', AMDGT: 0.750, OurImproved: 0.810 },
];

export default function UserPortal() {
  const [darkMode, setDarkMode] = useState(false);
  const [isLogged, setIsLogged] = useState(false);
  const [activeTab, setActiveTab] = useState('home'); // 'home', 'predict', 'multi-analyze'
  const [chartDataset, setChartDataset] = useState('B-dataset'); // Toggle chart
  
  // Data States
  const [datasetStats, setDatasetStats] = useState(null);
  const [nodeList, setNodeList] = useState({ drugs: [], diseases: [] }); // REAL DATA
  
  // Predict States
  const [searchMode, setSearchMode] = useState('drug2disease'); // 'drug2disease' | 'disease2drug'
  const [searchQuery, setSearchQuery] = useState('');
  const [showAutoComplete, setShowAutoComplete] = useState(false);
  const [topK, setTopK] = useState(15);
  const [datasetName, setDatasetName] = useState('C-dataset');
  
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState([]);
  
  // Graph States
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [selectedNode, setSelectedNode] = useState(null);
  const fgRef = useRef(); // Reference to stop jumping graph
  const fgRefMulti = useRef(); // Reference for multi analyze graph
  
  // Custom Node & Hover States
  const imageCache = useRef(new Map());
  const [hoverNode, setHoverNode] = useState(null);
  const [highlightNodes, setHighlightNodes] = useState(new Set());
  const [highlightLinks, setHighlightLinks] = useState(new Set());

  // Multi Predict States
  const [multiDrugsStr, setMultiDrugsStr] = useState('Aspirin, Ibuprofen');
  const [multiDiseasesStr, setMultiDiseasesStr] = useState('Hypertension, Migraine');
  const [multiThreshold, setMultiThreshold] = useState(0.8);
  
  // History
  const [history, setHistory] = useState([]);

  // Load Initial Stats
  useEffect(() => {
    fetch('http://127.0.0.1:8000/stats')
      .then(r => r.json())
      .then(data => setDatasetStats(data))
      .catch(e => console.error(e));
  }, []);

  // Fetch Nodes for AutoComplete based on Dataset
  useEffect(() => {
    fetch(`http://127.0.0.1:8000/nodes?dataset_name=${datasetName}`)
      .then(r => r.json())
      .then(data => setNodeList(data))
      .catch(e => console.error(e));
  }, [datasetName]);

  // Theme
  useEffect(() => {
    const root = window.document.documentElement;
    if (darkMode) root.classList.add('dark');
    else root.classList.remove('dark');
  }, [darkMode]);

  // Graph Build Helper
  const buildGraphFromResults = (sourceName, resultsArray, isSourceDrug) => {
    const nodesMap = new Map();
    nodesMap.set(sourceName, { id: sourceName, name: sourceName, group: isSourceDrug ? 'drug' : 'disease', val: 20 });
    
    const links = [];
    resultsArray.forEach(res => {
      const targetName = res.target;
      if (!nodesMap.has(targetName)) {
        nodesMap.set(targetName, { id: targetName, name: targetName, group: isSourceDrug ? 'disease' : 'drug', val: res.score > 0.8 ? 15 : 10 });
      }
      links.push({
        source: sourceName, target: targetName, value: res.score,
        color: res.score > 0.9 ? '#10b981' : (res.score > 0.5 ? '#f59e0b' : '#3b82f6')
      });
    });
    
    setGraphData({ nodes: Array.from(nodesMap.values()), links });
  };

  // Build Multi Graph Helper
  const buildMultiGraph = (resultsArray) => {
    const nodesMap = new Map();
    const links = [];
    resultsArray.forEach(res => {
      if (!nodesMap.has(res.source)) nodesMap.set(res.source, { id: res.source, name: res.source, group: 'drug', val: 15 });
      if (!nodesMap.has(res.target)) nodesMap.set(res.target, { id: res.target, name: res.target, group: 'disease', val: 15 });
      links.push({ source: res.source, target: res.target, value: res.score, color: res.score > 0.9 ? '#10b981' : '#f59e0b' });
    });
    setGraphData({ nodes: Array.from(nodesMap.values()), links });
  };

  // Prediction Submissions
  const handlePredict = async () => {
    if (!searchQuery) return;
    setIsLoading(true);
    setShowAutoComplete(false);
    setSelectedNode(null);
    try {
      const url = `http://127.0.0.1:8000/predict?query=${encodeURIComponent(searchQuery)}&mode=${searchMode}&top_k=${topK}&dataset_name=${datasetName}`;
      const response = await fetch(url);
      if (!response.ok) throw new Error("API Fails");
      const data = await response.json();
      setResults(data.results);
      
      const exactQuery = data.results.length > 0 ? data.results[0].source : searchQuery;
      buildGraphFromResults(exactQuery, data.results, searchMode === 'drug2disease');
      
      setHistory(prev => [{ id: Date.now(), query: exactQuery, mode: searchMode, time: 'Vừa xong', topK }, ...prev]);
    } catch (error) {
      alert(`Backend may be offline: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleMultiPredict = async () => {
    setIsLoading(true);
    setSelectedNode(null);
    try {
      const payload = {
        drugs: multiDrugsStr.split(',').map(s=>s.trim()),
        diseases: multiDiseasesStr.split(',').map(s=>s.trim()),
        dataset_name: datasetName,
        threshold: multiThreshold
      };
      const res = await fetch(`http://127.0.0.1:8000/predict_multi`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error("Multi predict failed.");
      const data = await res.json();
      setResults(data.results);
      buildMultiGraph(data.results);
    } catch (e) {
      console.error(e);
      alert("Backend error.");
    } finally {
      setIsLoading(false);
    }
  };

  // --- Graph Hover & Canvas Painting Logic ---
  const handleNodeHover = useCallback((node) => {
    const newHighlightNodes = new Set();
    const newHighlightLinks = new Set();
    if (node) {
      newHighlightNodes.add(node.id);
      graphData.links.forEach(link => {
        const s = typeof link.source === 'object' ? link.source.id : link.source;
        const t = typeof link.target === 'object' ? link.target.id : link.target;
        if (s === node.id || t === node.id) {
          newHighlightLinks.add(link);
          newHighlightNodes.add(s);
          newHighlightNodes.add(t);
        }
      });
    }
    setHoverNode(node);
    setHighlightNodes(newHighlightNodes);
    setHighlightLinks(newHighlightLinks);
  }, [graphData]);

  const paintNode = useCallback((node, ctx, globalScale) => {
    try {
      if (!node) return;
      if (typeof node.x !== 'number' || typeof node.y !== 'number') return;
      
      const isHovered = hoverNode === node;
      const isHighlighted = highlightNodes.has(node?.id);
      const isDimmed = hoverNode && !isHighlighted;
      const n_id = String(node.id || "");
      const n_name = String(node.name || "Unknown");
      const n_val = node.val || 10;

      ctx.save();
      ctx.globalAlpha = isDimmed ? 0.15 : 1;

      if (node.group === 'drug') {
        const size = Math.max(2, n_val);
        let img = imageCache.current.get(n_id);

        if (img === undefined) {
          imageCache.current.set(n_id, 'loading');
          const newImg = new Image();
          const extractName = n_id.replace('Drug_', ''); 
          newImg.src = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${encodeURIComponent(extractName)}/PNG`;
          newImg.onload = () => { 
             imageCache.current.set(n_id, newImg); 
             if (fgRef.current) {
                // Manually trigger a canvas tick frame safely!
                const currentZoom = fgRef.current.zoom();
                fgRef.current.zoom(currentZoom + 0.00001);
                setTimeout(() => fgRef.current.zoom(currentZoom), 0);
             }
          };
          newImg.onerror = () => { imageCache.current.set(n_id, 'error'); };
        }

        ctx.beginPath();
        ctx.arc(node.x, node.y, size / 2, 0, 2 * Math.PI);
        ctx.fillStyle = '#f0fdfa';
        ctx.fill();

        if (img && img.src && img.complete && img.naturalWidth > 0) {
          ctx.save();
          ctx.beginPath();
          ctx.arc(node.x, node.y, size / 2, 0, 2 * Math.PI);
          ctx.clip(); 
          ctx.drawImage(img, node.x - size / 2, node.y - size / 2, size, size);
          ctx.restore();
        } else {
          ctx.fillStyle = '#0d9488';
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(node.x, node.y, size / 2 + 1, 0, 2 * Math.PI);
        ctx.lineWidth = isHovered ? 2 / globalScale : 1.5;
        ctx.strokeStyle = isHighlighted ? '#f59e0b' : '#94a3b8';
        ctx.stroke();

      } else {
        const size = Math.max(2, n_val * 0.8);
        const gradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, size / 2);
        gradient.addColorStop(0, '#f43f5e'); 
        gradient.addColorStop(1, '#9f1239'); 
        
        ctx.beginPath();
        ctx.arc(node.x, node.y, size / 2, 0, 2 * Math.PI);
        ctx.fillStyle = gradient;
        ctx.fill();

        ctx.lineWidth = isHovered ? 2 / globalScale : 1;
        ctx.strokeStyle = isHighlighted ? '#fda4af' : '#4c0519';
        ctx.stroke();
      }

      if (globalScale > 2 || isHighlighted || isHovered) {
        const fontSize = isHovered ? 14 / globalScale : 12 / globalScale;
        ctx.font = `${fontSize}px Sans-Serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = darkMode ? '#cbd5e1' : '#334155';
        const labelY = node.y + (node.group === 'drug' ? n_val : n_val * 0.8) / 2 + 6 / globalScale + fontSize / 2;
        
        const textWidth = ctx.measureText(n_name).width;
        ctx.fillStyle = darkMode ? 'rgba(15, 23, 42, 0.7)' : 'rgba(255, 255, 255, 0.7)';
        ctx.fillRect(node.x - textWidth/2 - 2/globalScale, labelY - fontSize/2 - 2/globalScale, textWidth + 4/globalScale, fontSize + 4/globalScale);
        
        ctx.fillStyle = darkMode ? '#e2e8f0' : '#0f172a';
        ctx.fillText(n_name, node.x, labelY);
      }
      
      ctx.restore();
    } catch (e) {
      console.error(e);
      ctx.restore();
    }
  }, [hoverNode, highlightNodes, darkMode]);

  return (
    <div className="min-h-screen flex font-sans bg-slate-50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 selection:bg-teal-500/30">
      
      {/* ----------------- LEFT SIDEBAR (FIXED) ----------------- */}
      <aside className="w-64 border-r border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl flex flex-col fixed h-full z-40">
        <div className="p-6 flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-teal-600 flex items-center justify-center text-white shadow"><Network size={18}/></div>
          <h1 className="font-bold text-lg text-teal-700 dark:text-teal-400">AMNTDDA</h1>
        </div>
        
        <nav className="flex-1 px-4 space-y-2 overflow-y-auto">
          <button onClick={() => {setActiveTab('home'); setGraphData({nodes:[], links:[]}); setResults([])}} className={cn("w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all font-medium", activeTab === 'home' ? "bg-teal-50 dark:bg-teal-900/20 text-teal-700 dark:text-teal-300" : "hover:bg-slate-100 dark:hover:bg-slate-800/50 text-slate-600 dark:text-slate-400")}>
            <HomeIcon size={18}/> Trang Chủ
          </button>
          <button onClick={() => {setActiveTab('predict'); setGraphData({nodes:[], links:[]}); setResults([])}} className={cn("w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all font-medium", activeTab === 'predict' ? "bg-teal-50 dark:bg-teal-900/20 text-teal-700 dark:text-teal-300" : "hover:bg-slate-100 dark:hover:bg-slate-800/50 text-slate-600 dark:text-slate-400")}>
            <Search size={18}/> Dự Đoán Đơn
          </button>
          <button onClick={() => {setActiveTab('multi'); setGraphData({nodes:[], links:[]}); setResults([])}} className={cn("w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all font-medium", activeTab === 'multi' ? "bg-teal-50 dark:bg-teal-900/20 text-teal-700 dark:text-teal-300" : "hover:bg-slate-100 dark:hover:bg-slate-800/50 text-slate-600 dark:text-slate-400")}>
            <Network size={18}/> Phân Tích Đa Đối Tượng
          </button>
          
          <div className="pt-6 pb-2">
            <p className="px-4 text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Lịch sử tra cứu</p>
            {history.map(item => (
              <div key={item.id} className="group relative px-4 py-2 hover:bg-slate-100 dark:hover:bg-slate-800/50 rounded-xl cursor-pointer">
                <p className="text-sm font-medium text-slate-700 dark:text-slate-300 truncate">{item.query}</p>
                <p className="text-[10px] text-slate-400 mt-0.5">{item.time}</p>
              </div>
            ))}
          </div>
        </nav>

        <div className="p-4 mt-auto border-t border-slate-200 dark:border-slate-800">
          <button onClick={() => setDarkMode(!darkMode)} className="w-full flex items-center gap-3 px-4 py-2.5 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800/50 transition-colors text-sm font-medium">
            {darkMode ? <Sun size={18}/> : <Moon size={18}/>}
            {darkMode ? 'Light Mode' : 'Dark Mode'}
          </button>
        </div>
      </aside>

      {/* ----------------- MAIN CONTENT AREA ----------------- */}
      <main className="ml-64 flex-1 h-screen overflow-hidden flex flex-col">
        
        {/* ========================================================= */}
        {/* VIEW: TRANG CHỦ (HOME) */}
        {/* ========================================================= */}
        {activeTab === 'home' && (
          <div className="p-8 overflow-y-auto w-full h-full">
            <h2 className="text-3xl font-bold mb-8 text-slate-800 dark:text-white">Tổng Quan Dự Án</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
              {['B-dataset', 'C-dataset', 'F-dataset'].map(ds => {
                const stats = datasetStats?.[ds] || { drugs: 0, diseases: 0, ddas: 0, sparsity: '0%' };
                return (
                  <div key={ds} className="p-6 bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 shadow-sm hover:shadow-md transition-shadow">
                    <h3 className="text-xl font-bold text-teal-600 dark:text-teal-400 mb-4">{ds}</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between"><span className="text-slate-500">Thuốc (Drugs):</span> <span className="font-semibold">{stats.drugs}</span></div>
                      <div className="flex justify-between"><span className="text-slate-500">Bệnh (Diseases):</span> <span className="font-semibold">{stats.diseases}</span></div>
                      <div className="flex justify-between"><span className="text-slate-500">Liên kết DDAs:</span> <span className="font-semibold">{stats.ddas}</span></div>
                      <div className="flex justify-between"><span className="text-slate-500">Mức thưa (Sparsity):</span> <span className="font-semibold">{stats.sparsity}</span></div>
                    </div>
                  </div>
                )
              })}
            </div>

            <div className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 p-8 shadow-sm">
              <div className="flex items-center justify-between mb-6">
                 <h3 className="text-xl font-bold flex items-center gap-2">
                   <BarChart2 className="text-teal-500" /> So Sánh Hiệu Suất Mô Hình
                 </h3>
                 <select value={chartDataset} onChange={e => setChartDataset(e.target.value)} className="bg-slate-100 dark:bg-slate-800 border-none px-4 py-2 rounded-lg font-bold outline-none text-teal-600">
                    <option value="B-dataset">B-dataset</option>
                    <option value="C-dataset">C-dataset</option>
                    <option value="F-dataset">F-dataset</option>
                 </select>
              </div>
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartDataset === 'B-dataset' ? performanceDataB : (chartDataset === 'C-dataset' ? performanceDataC : performanceDataF)} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                    <XAxis dataKey="metric" tick={{fill: darkMode ? '#94a3b8' : '#475569'}} />
                    <YAxis domain={chartDataset === 'B-dataset' ? [0.6, 1.0] : [0.5, 1.0]} tick={{fill: darkMode ? '#94a3b8' : '#475569'}} />
                    <RechartsTooltip cursor={{fill: darkMode ? '#334155' : '#f1f5f9'}} contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}} />
                    <Legend />
                    <Bar dataKey="AMDGT" name="AMDGT Gốc (KBS 2024)" fill="#94a3b8" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="OurImproved" name="AMNTDDA (Đề xuất)" fill="#0d9488" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* VIEW: DỰ ĐOÁN ĐƠN (LEFT GRAPH / RIGHT PANEL) */}
        {/* ========================================================= */}
        {activeTab === 'predict' && (
          <div className="flex w-full flex-1 min-h-0">
            
            {/* LEFT 35%: GRAPH VISUALIZATION */}
            <div className="w-[35%] shrink-0 overflow-hidden bg-slate-100 dark:bg-black/20 border-r border-slate-200 dark:border-slate-800 relative flex flex-col">
              <div className="absolute top-4 left-4 z-10 bg-white/70 dark:bg-slate-900/70 backdrop-blur px-4 py-2 rounded-xl text-sm font-semibold shadow-sm">
                Đồ thị Không gian
              </div>
              
              {/* Force Graph Container */}
              <div className="flex-1 w-full h-full relative cursor-move overflow-hidden">
                {graphData.nodes.length > 0 ? (
                  <ForceGraph2D
                    ref={fgRef}
                    graphData={graphData}
                    nodeLabel=""
                    nodeCanvasObject={paintNode}
                    linkColor={link => highlightLinks.has(link) ? '#14b8a6' : (hoverNode ? (darkMode ? 'rgba(148, 163, 184, 0.1)' : 'rgba(148, 163, 184, 0.2)') : link.color)}
                    linkWidth={link => highlightLinks.has(link) ? 3 : 1}
                    onNodeHover={handleNodeHover}
                    onNodeClick={(node) => {
                       if (!node) return;
                       setSelectedNode(node);
                    }}
                    nodeRelSize={2}
                    d3AlphaDecay={0.05}
                    d3VelocityDecay={0.6}
                    cooldownTicks={50}
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-slate-400">
                    Chưa có dữ liệu đồ thị. Hãy chạy AI dự đoán.
                  </div>
                )}
              </div>
              
              {/* Selected Node Details & Molecular Structure */}
              {selectedNode && (
                <div className="h-48 border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 shrink-0 flex gap-4 animate-in slide-in-from-bottom">
                  <div className="flex-1">
                    <h4 className="font-bold text-lg mb-1">{selectedNode.name}</h4>
                    <span className={cn("px-2 py-0.5 text-xs font-bold rounded uppercase", selectedNode.group === 'drug' ? 'bg-teal-100 text-teal-700' : 'bg-rose-100 text-rose-700')}>
                      {selectedNode.group}
                    </span>
                    <p className="mt-2 text-xs text-slate-500">Click graph background to close.</p>
                  </div>
                  {/* Molecular image from PubChem (only for drugs, simple generic lookup by name) */}
                  {selectedNode.group === 'drug' && (
                    <div className="w-32 h-32 rounded-lg bg-white overflow-hidden border border-slate-200 shrink-0 flex items-center justify-center p-2 relative group">
                       <img 
                          src={`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${encodeURIComponent(selectedNode.name.replace('Drug_',''))}/PNG`} 
                          alt="Structure"
                          onError={(e) => { e.target.style.display='none'; e.target.nextSibling.style.display='block'; }}
                          className="w-full h-full object-contain"
                       />
                       <span className="hidden text-xs text-slate-400 text-center">No structure found</span>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* RIGHT 65%: PREDICTIONS PANEL */}
            <div className="w-[65%] shrink-0 p-8 overflow-y-auto">
              <div className="max-w-4xl mx-auto space-y-6">
                <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl p-6 shadow-sm">
                  <h2 className="text-2xl font-bold mb-6 text-slate-800 dark:text-slate-100">Cấu hình Dự đoán</h2>
                  
                  <div className="flex gap-4 mb-6">
                    <select value={datasetName} onChange={e => setDatasetName(e.target.value)} className="bg-slate-100 dark:bg-slate-800 border-none px-4 py-2.5 rounded-xl font-semibold outline-none">
                      <option value="C-dataset">C-dataset</option><option value="B-dataset">B-dataset</option><option value="F-dataset">F-dataset</option>
                    </select>
                    
                    <div className="flex bg-slate-100 dark:bg-slate-800 p-1 rounded-xl w-full">
                      <button onClick={() => setSearchMode('drug2disease')} className={cn("flex-1 py-1.5 text-sm font-semibold rounded-lg transition-all", searchMode === 'drug2disease' ? "bg-white dark:bg-slate-600 shadow" : "text-slate-500")}>Thuốc ➔ Bệnh</button>
                      <button onClick={() => setSearchMode('disease2drug')} className={cn("flex-1 py-1.5 text-sm font-semibold rounded-lg transition-all", searchMode === 'disease2drug' ? "bg-white dark:bg-slate-600 shadow" : "text-slate-500")}>Bệnh ➔ Thuốc</button>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-bold mb-2">{searchMode === 'drug2disease' ? 'Tên Thuốc (vd: Aspirin)' : 'Tên Bệnh (vd: Hypertension)'}</label>
                      <div className="relative">
                        <Search className="absolute left-4 top-3.5 h-5 w-5 text-slate-400" />
                        <input type="text" list="node-suggestions" value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} className="w-full pl-12 pr-4 py-3 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-2xl focus:ring-2 focus:ring-teal-500 font-medium" placeholder="Nhập từ khóa..."/>
                        <datalist id="node-suggestions">
                          {(searchMode === 'drug2disease' ? (nodeList?.drugs || []) : (nodeList?.diseases || [])).slice(0, 100).map((n, i) => <option key={i} value={n} />)}
                        </datalist>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between mb-1"><label className="text-sm font-semibold">Top K ({topK})</label></div>
                      <input type="range" min="5" max="50" value={topK} onChange={(e) => setTopK(parseInt(e.target.value))} className="w-full accent-teal-600"/>
                    </div>

                    <button onClick={handlePredict} disabled={!searchQuery || isLoading} className="w-full py-4 bg-teal-600 hover:bg-teal-700 disabled:opacity-50 text-white rounded-xl font-bold text-lg flex items-center justify-center gap-2 shadow-lg shadow-teal-500/20 active:scale-95 transition-all">
                      {isLoading ? "Đang xử lý..." : "Chạy Dự Đoán AMNTDDA"}
                    </button>
                  </div>
                </div>

                {/* RESULTS TABLE */}
                {results.length > 0 && !isLoading && (
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl p-6 shadow-sm">
                    <h3 className="text-lg font-bold mb-4">Kết Quả (Top {results.length})</h3>
                    <table className="w-full text-left border-collapse text-sm">
                      <thead>
                        <tr className="bg-slate-50 dark:bg-slate-800/80 uppercase text-slate-500">
                          <th className="p-3">Hạng</th><th className="p-3">Tên Mục Tiêu</th><th className="p-3">Độ tin cậy</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.map((r, i) => (
                           <tr key={i} className="border-b border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/30">
                              <td className="p-3 font-mono">#{i + 1}</td>
                              <td className="p-3 font-medium">{r.target}</td>
                              <td className="p-3"><span className="font-bold text-teal-600 dark:text-teal-400">{(r.score * 100).toFixed(2)}%</span></td>
                           </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* VIEW: PHÂN TÍCH ĐA ĐỐI TƯỢNG */}
        {/* ========================================================= */}
        {activeTab === 'multi' && (
          <div className="flex w-full flex-1 min-h-0">
            <div className="w-[35%] shrink-0 overflow-hidden bg-slate-100 dark:bg-black/20 border-r border-slate-200 dark:border-slate-800 relative flex flex-col">
               <div className="absolute top-4 left-4 z-10 bg-white/70 dark:bg-slate-900/70 backdrop-blur px-4 py-2 rounded-xl text-sm font-semibold">Đồ thị Mạng lưới Đa tương tác</div>
               <div className="flex-1 w-full h-full relative cursor-move overflow-hidden">
                {graphData.nodes.length > 0 ? (
                  <ForceGraph2D
                    ref={fgRefMulti}
                    graphData={graphData}
                    nodeLabel=""
                    nodeCanvasObject={paintNode}
                    linkColor={link => highlightLinks.has(link) ? '#14b8a6' : (hoverNode ? (darkMode ? 'rgba(148, 163, 184, 0.1)' : 'rgba(148, 163, 184, 0.2)') : link.color)}
                    linkWidth={link => highlightLinks.has(link) ? 3 : 1.5}
                    onNodeHover={handleNodeHover}
                    nodeRelSize={2}
                    d3AlphaDecay={0.05}
                    d3VelocityDecay={0.6}
                    cooldownTicks={50}
                  />
                ) : <div className="w-full h-full flex justify-center items-center text-slate-400">Phân tích để vẽ đồ thị</div>}
               </div>
            </div>
            <div className="w-[65%] shrink-0 p-8 overflow-y-auto">
              <div className="max-w-4xl mx-auto space-y-6">
                <div className="bg-white dark:bg-slate-900 rounded-3xl p-6 border border-slate-200 dark:border-slate-800">
                  <h2 className="text-2xl font-bold mb-6">Phân Tích Many-to-Many</h2>
                  
                  <div className="space-y-4">
                     <div>
                       <label className="block text-sm font-bold mb-2">Danh sách Thuốc (cách nhau dấu phẩy)</label>
                       <textarea rows="2" value={multiDrugsStr} onChange={e=>setMultiDrugsStr(e.target.value)} className="w-full p-3 bg-slate-50 dark:bg-slate-950 rounded-xl border border-slate-200 dark:border-slate-800" placeholder="Aspirin, Paracetamol..."/>
                     </div>
                     <div>
                       <label className="block text-sm font-bold mb-2">Danh sách Bệnh (cách nhau dấu phẩy)</label>
                       <textarea rows="2" value={multiDiseasesStr} onChange={e=>setMultiDiseasesStr(e.target.value)} className="w-full p-3 bg-slate-50 dark:bg-slate-950 rounded-xl border border-slate-200 dark:border-slate-800" placeholder="Hypertension, Asthma..."/>
                     </div>
                     <div className="flex gap-4">
                        <select value={datasetName} onChange={e => setDatasetName(e.target.value)} className="bg-slate-100 dark:bg-slate-800 rounded-xl px-4 flex-1">
                          <option value="B-dataset">B-dataset</option>
                          <option value="C-dataset">C-dataset</option>
                          <option value="F-dataset">F-dataset</option>
                        </select>
                        <div className="flex-1">
                          <label className="text-sm font-semibold">Ngưỡng Threshold: {multiThreshold}</label>
                          <input type="range" min="0.1" max="0.99" step="0.01" value={multiThreshold} onChange={e=>setMultiThreshold(parseFloat(e.target.value))} className="w-full accent-teal-600"/>
                        </div>
                     </div>
                     <button onClick={handleMultiPredict} disabled={isLoading} className="w-full py-4 bg-teal-600 text-white rounded-xl font-bold">
                        {isLoading ? "Đang xử lý O(N*M)..." : "Phân Tích Chéo"}
                     </button>
                  </div>
                </div>

                {results.length > 0 && !isLoading && (
                  <div className="bg-white dark:bg-slate-900 rounded-3xl p-6 border border-slate-200 dark:border-slate-800">
                    <h3 className="text-lg font-bold mb-4">Các liên kết đạt ngưỡng ({results.length})</h3>
                    <div className="grid grid-cols-2 gap-4">
                      {results.map((r, i) => (
                        <div key={i} className="p-3 bg-slate-50 dark:bg-slate-800 rounded-xl flex justify-between items-center text-sm">
                           <span className="truncate flex-1 font-medium">{r.source}</span>
                           <ArrowRight size={14} className="mx-2 text-slate-400"/>
                           <span className="truncate flex-1 font-medium">{r.target}</span>
                           <span className="ml-2 font-bold text-teal-600">{(r.score*100).toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}
