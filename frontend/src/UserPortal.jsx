import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Search, Moon, Sun, Network, ArrowRight, Home as HomeIcon, CheckSquare, BarChart2, Settings, Layers, Shuffle, X, Database, Info, FileText, PlayCircle, Clipboard
} from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import ForceGraph2D from 'react-force-graph-2d';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

// PERFORMANCE MOCK DATA
const performanceDataB = [
  { metric: 'AUC', Original: 0.932, Improved: 0.965 },
  { metric: 'AUPR', Original: 0.915, Improved: 0.952 },
  { metric: 'F1', Original: 0.860, Improved: 0.901 },
  { metric: 'MCC', Original: 0.810, Improved: 0.870 },
];
const performanceDataC = [
  { metric: 'AUC', Original: 0.825, Improved: 0.875 },
  { metric: 'AUPR', Original: 0.812, Improved: 0.854 },
  { metric: 'F1', Original: 0.760, Improved: 0.801 },
  { metric: 'MCC', Original: 0.680, Improved: 0.720 },
];
const performanceDataF = [
  { metric: 'AUC', Original: 0.880, Improved: 0.920 },
  { metric: 'AUPR', Original: 0.850, Improved: 0.895 },
  { metric: 'F1', Original: 0.820, Improved: 0.865 },
  { metric: 'MCC', Original: 0.750, Improved: 0.810 },
];

export default function UserPortal() {
  const [darkMode, setDarkMode] = useState(false);
  const [activeTab, setActiveTab] = useState('home'); 
  const [datasetName, setDatasetName] = useState('C-dataset');
  
  // Data States
  const [datasetStats, setDatasetStats] = useState(null);
  const [nodeList, setNodeList] = useState({ drugs: [], diseases: [] });
  const [hyperData, setHyperData] = useState({ params: {}, metrics: [] });
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState([]);

  // Control States
  const [searchMode, setSearchMode] = useState('drug2disease');
  const [searchQuery, setSearchQuery] = useState('');
  const [topK, setTopK] = useState(15);
  const [multiDrugsStr, setMultiDrugsStr] = useState('Aspirin, Ibuprofen');
  const [multiDiseasesStr, setMultiDiseasesStr] = useState('Hypertension, Migraine');
  const [multiThreshold, setMultiThreshold] = useState(0.8);
  const [nRandomDrugs, setNRandomDrugs] = useState(5);
  const [nRandomDiseases, setNRandomDiseases] = useState(5);

  // Graph States & Refs
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoverNode, setHoverNode] = useState(null);
  const [highlightNodes, setHighlightNodes] = useState(new Set());
  const [highlightLinks, setHighlightLinks] = useState(new Set());
  
  const fgPredict = useRef();
  const fgRandom = useRef();
  const fgMulti = useRef();
  const imageCache = useRef(new Map());

  // --- API FETCH ---
  useEffect(() => {
    fetch('http://127.0.0.1:8000/stats').then(r => r.json()).then(d => setDatasetStats(d)).catch(e => console.error(e));
  }, []);

  useEffect(() => {
    setIsLoading(true);
    const p1 = fetch(`http://127.0.0.1:8000/nodes?dataset_name=${datasetName}`).then(r => r.json());
    const p2 = fetch(`http://127.0.0.1:8000/hyperparameters?dataset_name=${datasetName}`).then(r => r.json());
    Promise.all([p1, p2]).then(([n, h]) => {
      setNodeList(n || { drugs: [], diseases: [] });
      setHyperData(h || { params: {}, metrics: [] });
    }).finally(() => setIsLoading(false));
  }, [datasetName]);

  useEffect(() => {
    const root = window.document.documentElement;
    if (darkMode) root.classList.add('dark'); else root.classList.remove('dark');
  }, [darkMode]);

  // --- GRAPH BUILDERS ---
  const buildPredictGraph = (sName, sId, sSmiles, resArr, isDrug) => {
    const nodes = new Map();
    nodes.set(sName, { id: sName, name: sName, realId: sId, smiles: sSmiles, group: isDrug?'drug':'disease', val: 24 });
    resArr.forEach(r => {
      if (!nodes.has(r.target)) nodes.set(r.target, { id: r.target, name: r.target, realId: r.target_id, smiles: r.target_smiles || "", group: isDrug?'disease':'drug', val: 16 });
    });
    setGraphData({ nodes: Array.from(nodes.values()), links: resArr.map(r=>({ source: sName, target: r.target, value: r.score, color: r.score>0.8?'#0d9488':'#94a3b8' })) });
  };

  const buildMultiGraph = (resArr) => {
    const nodes = new Map();
    resArr.forEach(r => {
      if (!nodes.has(r.source)) nodes.set(r.source, { id: r.source, name: r.source, group: 'drug', val: 18, smiles: r.source_smiles || "", realId: r.source_id });
      if (!nodes.has(r.target)) nodes.set(r.target, { id: r.target, name: r.target, group: 'disease', val: 16, realId: r.target_id });
    });
    setGraphData({ nodes: Array.from(nodes.values()), links: resArr.map(r=>({ source: r.source, target: r.target, value: r.score, color: r.score>0.8?'#0d9488':'#f59e0b' })) });
  };

  // --- HANDLERS ---
  const handlePredict = async () => {
    if (!searchQuery) return; setIsLoading(true); setSelectedNode(null); setResults([]);
    try {
      const r = await fetch(`http://127.0.0.1:8000/predict?query=${encodeURIComponent(searchQuery)}&mode=${searchMode}&top_k=${topK}&dataset_name=${datasetName}`);
      const d = await r.json(); setResults(d.results || []);
      if (d.results?.length) buildPredictGraph(d.results[0].source, d.results[0].source_id, d.results[0].source_smiles, d.results, searchMode==='drug2disease');
    } catch(e) { console.error(e); } finally { setIsLoading(false); }
  };

  const handleRandomPredict = async () => {
    setIsLoading(true); setSelectedNode(null); setResults([]);
    try {
      const nR = await fetch(`http://127.0.0.1:8000/random_nodes?n_drugs=${nRandomDrugs}&n_diseases=${nRandomDiseases}&dataset_name=${datasetName}`);
      const nD = await nR.json();
      const pR = await fetch(`http://127.0.0.1:8000/predict_multi`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ drugs: nD.drugs, diseases: nD.diseases, dataset_name: datasetName, threshold: 0.4 })
      });
      const d = await pR.json(); setResults(d.results || []); buildMultiGraph(d.results || []);
    } catch(e) { console.error(e); } finally { setIsLoading(false); }
  };

  const handleMultiPredict = async () => {
    setIsLoading(true); setSelectedNode(null); setResults([]);
    try {
      const pR = await fetch(`http://127.0.0.1:8000/predict_multi`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ drugs: multiDrugsStr.split(',').map(s=>s.trim()), diseases: multiDiseasesStr.split(',').map(s=>s.trim()), dataset_name: datasetName, threshold: multiThreshold })
      });
      const d = await pR.json(); setResults(d.results || []); buildMultiGraph(d.results || []);
    } catch(e) { console.error(e); } finally { setIsLoading(false); }
  };

  // --- GRAPH RENDERING ---
  const handleHover = useCallback((n) => {
    const nH = new Set(); const lH = new Set();
    if (n) {
      nH.add(n.id);
      graphData.links.forEach(l => {
        const s = typeof l.source==='object'?l.source.id:l.source;
        const t = typeof l.target==='object'?l.target.id:l.target;
        if (s===n.id || t===n.id) { lH.add(l); nH.add(s); nH.add(t); }
      });
    }
    setHoverNode(n); setHighlightNodes(nH); setHighlightLinks(lH);
  }, [graphData]);

  const paintNode = useCallback((n, ctx, scale) => {
    const isHi = highlightNodes.has(n.id); const isDim = hoverNode && !isHi;
    ctx.save(); ctx.globalAlpha = isDim?0.15:1;
    const size = n.val || 10;
    if (n.group==='drug') {
      ctx.beginPath(); ctx.arc(n.x, n.y, size/2, 0, 2*Math.PI); 
      ctx.fillStyle='#0d9488'; ctx.fill();
      ctx.strokeStyle = isHi?'#f59e0b':'#14b8a6'; ctx.lineWidth=isHi?2:1; ctx.stroke();
    } else {
      ctx.beginPath(); ctx.arc(n.x, n.y, (size*0.7)/2, 0, 2*Math.PI); ctx.fillStyle='#e11d48'; ctx.fill();
      ctx.strokeStyle = isHi?'#fb7185':'#881337'; ctx.lineWidth=isHi?2:0.5; ctx.stroke();
    }
    if (scale>2 || isHi) {
      const fs=12/scale; ctx.font=`${fs}px Sans-Serif`; ctx.textAlign='center'; ctx.fillStyle=darkMode?'#e2e8f0':'#0f172a';
      ctx.fillText(n.name, n.x, n.y+size/2+5/scale+fs/2);
    }
    ctx.restore();
  }, [hoverNode, highlightNodes, darkMode]);

  return (
    <div className="flex h-screen w-screen bg-slate-50 dark:bg-slate-950 font-sans text-slate-800 dark:text-slate-100 overflow-hidden">
      {/* SIDEBAR */}
      <aside className="w-64 flex-shrink-0 flex flex-col bg-white dark:bg-slate-900 border-r dark:border-slate-800">
        <div className="p-8 flex items-center gap-3">
          <div className="w-10 h-10 bg-teal-600 rounded-xl flex items-center justify-center text-white shadow-lg"><Network size={24}/></div>
          <h1 className="font-black text-xl text-teal-700 dark:text-teal-400 uppercase tracking-tighter">AMNTDDA</h1>
        </div>
        <nav className="flex-1 px-4 space-y-1">
          <NavItem active={activeTab==='home'} icon={<HomeIcon size={18}/>} label="Trang Chủ" onClick={()=>setActiveTab('home')}/>
          <NavItem active={activeTab==='predict'} icon={<Search size={18}/>} label="Dự Đoán Đơn" onClick={()=>setActiveTab('predict')}/>
          <NavItem active={activeTab==='random'} icon={<Shuffle size={18}/>} label="Dự Đoán Ngẫu Nhiên" onClick={()=>setActiveTab('random')}/>
          <NavItem active={activeTab==='multi'} icon={<Layers size={18}/>} label="Many-to-Many" onClick={()=>setActiveTab('multi')}/>
          <NavItem active={activeTab==='hyperparams'} icon={<Settings size={18}/>} label="Siêu Tham Số" onClick={()=>setActiveTab('hyperparams')}/>
          <NavItem active={activeTab==='comparison'} icon={<BarChart2 size={18}/>} label="So Sánh Mô Hình" onClick={()=>setActiveTab('comparison')}/>
        </nav>
        <div className="p-4 border-t dark:border-slate-800">
          <button onClick={()=>setDarkMode(!darkMode)} className="w-full flex items-center gap-3 px-4 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl transition-all">
            {darkMode?<Sun size={18} className="text-amber-500"/>:<Moon size={18}/>}
            <span className="text-xs font-black uppercase">{darkMode?'Sáng':'Tối'}</span>
          </button>
        </div>
      </aside>

      {/* MAIN VIEW */}
      <main className="flex-1 flex flex-col overflow-hidden relative">
        <header className="h-16 flex items-center justify-between px-8 bg-white/80 dark:bg-slate-900/80 backdrop-blur border-b dark:border-slate-800 shrink-0 z-20">
          <h2 className="font-black text-lg capitalize">{activeTab}</h2>
          <div className="flex items-center gap-4">
             <span className="text-[10px] font-black text-slate-400 uppercase">Dataset:</span>
             <select value={datasetName} onChange={e=>setDatasetName(e.target.value)} className="bg-slate-100 dark:bg-slate-800 px-4 py-1.5 rounded-xl font-bold text-teal-600 outline-none">
               <option value="C-dataset">C-dataset</option><option value="B-dataset">B-dataset</option><option value="F-dataset">F-dataset</option>
             </select>
          </div>
        </header>

        <div className="flex-1 overflow-hidden relative flex flex-col">
          {activeTab==='home' && (
            <div className="flex-1 overflow-y-auto p-8 space-y-8 animate-in fade-in">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {['B-dataset', 'C-dataset', 'F-dataset'].map(ds => (
                  <div key={ds} className="p-8 bg-white dark:bg-slate-900 border dark:border-slate-800 rounded-[32px] shadow-sm hover:border-teal-500 transition-all">
                    <div className="flex justify-between items-start mb-4"><h3 className="text-xl font-black text-teal-600 uppercase">{ds}</h3><Database size={24} className="text-slate-200"/></div>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between"><span>Thuốc:</span> <span className="font-bold">{datasetStats?.[ds]?.drugs || 0}</span></div>
                      <div className="flex justify-between"><span>Bệnh:</span> <span className="font-bold">{datasetStats?.[ds]?.diseases || 0}</span></div>
                    </div>
                  </div>
                ))}
              </div>
              <div className="bg-white dark:bg-slate-900 p-8 border dark:border-slate-800 rounded-[32px] shadow-sm">
                 <h3 className="text-xl font-black mb-8 flex items-center gap-3"><BarChart2 className="text-teal-500"/> Hiệu suất mô hình AMNTDDA</h3>
                 <div className="h-80 w-full">
                    <ResponsiveContainer><BarChart data={datasetName==='B-dataset'?performanceDataB:datasetName==='C-dataset'?performanceDataC:performanceDataF}><CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.1}/><XAxis dataKey="metric"/><YAxis domain={[0.6, 1.0]} /><RechartsTooltip contentStyle={{borderRadius:'16px',border:'none',boxShadow:'0 10px 15px rgb(0 0 0 / 0.1)'}}/><Legend /><Bar dataKey="Improved" name="AMNTDDA (Ours)" fill="#0d9488" radius={[8, 8, 0, 0]} /><Bar dataKey="Original" name="Baseline" fill="#94a3b8" radius={[8, 8, 0, 0]} /></BarChart></ResponsiveContainer>
                 </div>
              </div>
            </div>
          )}

          {activeTab==='predict' && (
            <div className="flex-1 flex w-full h-full overflow-hidden">
               <div className="w-0 flex-1 bg-slate-100 dark:bg-black/20 relative">
                  <ForceGraph2D ref={fgPredict} graphData={graphData} nodeCanvasObject={paintNode} onNodeClick={setSelectedNode} onNodeHover={handleHover} linkColor={l=>highlightLinks.has(l)?'#14b8a6':l.color} backgroundColor={darkMode?'#0f172a':'#f8fafc'} />
                  {selectedNode && <DetailDrawer node={selectedNode} onClose={()=>setSelectedNode(null)} />}
               </div>
               <div className="w-[380px] min-w-[380px] flex-shrink-0 flex flex-col bg-white dark:bg-slate-900 shadow-2xl z-30 border-l dark:border-slate-800 overflow-y-auto p-8 space-y-8">
                  <h3 className="text-xl font-black text-teal-600 flex items-center gap-2"><PlayCircle size={24}/> Dự Đoán</h3>
                  <div className="space-y-6">
                    <div className="flex bg-slate-100 dark:bg-slate-800 p-1 rounded-2xl">
                       <button onClick={()=>setSearchMode('drug2disease')} className={cn("flex-1 py-2 text-[10px] font-black rounded-xl", searchMode==='drug2disease'?"bg-white dark:bg-slate-600 shadow-sm text-teal-600":"text-slate-400")}>THUỐC</button>
                       <button onClick={()=>setSearchMode('disease2drug')} className={cn("flex-1 py-2 text-[10px] font-black rounded-xl", searchMode==='disease2drug'?"bg-white dark:bg-slate-600 shadow-sm text-teal-600":"text-slate-400")}>BỆNH</button>
                    </div>
                    <div className="space-y-2">
                       <label className="text-[10px] font-black text-slate-400 uppercase">Tên Tìm Kiếm</label>
                       <input type="text" list="predict-list" value={searchQuery} onChange={e=>setSearchQuery(e.target.value)} className="w-full p-4 bg-slate-50 dark:bg-slate-950 border dark:border-slate-800 rounded-2xl font-bold outline-none" placeholder="Nhập tên..."/>
                       <datalist id="predict-list">{(searchMode==='drug2disease'?nodeList.drugs:nodeList.diseases).slice(0, 100).map(n=><option key={n} value={n}/>)}</datalist>
                    </div>
                    <div className="space-y-2">
                       <div className="flex justify-between"><label className="text-[10px] font-black text-slate-400">Top-K</label><span className="font-black text-teal-600">{topK}</span></div>
                       <input type="range" min="5" max="50" value={topK} onChange={e=>setTopK(parseInt(e.target.value))} className="w-full accent-teal-600"/>
                    </div>
                    <button onClick={handlePredict} disabled={isLoading||!searchQuery} className="w-full py-5 bg-teal-600 text-white rounded-2xl font-black shadow-xl flex items-center justify-center gap-2">{isLoading?"ĐANG XỬ LÝ...":"DỰ ĐOÁN"}</button>
                  </div>
                  <ResultsList results={results} isLoading={isLoading} />
               </div>
            </div>
          )}

          {activeTab==='random' && (
            <div className="flex-1 flex w-full h-full overflow-hidden">
               <div className="w-0 flex-1 bg-slate-100 dark:bg-black/20 relative">
                  <ForceGraph2D ref={fgRandom} graphData={graphData} nodeCanvasObject={paintNode} onNodeClick={setSelectedNode} onNodeHover={handleHover} backgroundColor={darkMode?'#0f172a':'#f8fafc'}/>
                  {selectedNode && <DetailDrawer node={selectedNode} onClose={()=>setSelectedNode(null)} />}
               </div>
               <div className="w-[380px] min-w-[380px] flex-shrink-0 flex flex-col bg-white dark:bg-slate-900 shadow-2xl z-30 border-l dark:border-slate-800 overflow-y-auto p-8 space-y-8">
                  <h3 className="text-xl font-black text-teal-600 flex items-center gap-2"><Shuffle size={24}/> Ngẫu Nhiên</h3>
                  <div className="space-y-8">
                    <div className="space-y-4">
                      <div className="flex justify-between"><label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Drugs: {nRandomDrugs}</label></div>
                      <input type="range" min="2" max="10" value={nRandomDrugs} onChange={e=>setNRandomDrugs(parseInt(e.target.value))} className="w-full accent-teal-600"/>
                    </div>
                    <div className="space-y-4">
                      <div className="flex justify-between"><label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Diseases: {nRandomDiseases}</label></div>
                      <input type="range" min="2" max="10" value={nRandomDiseases} onChange={e=>setNRandomDiseases(parseInt(e.target.value))} className="w-full accent-teal-600"/>
                    </div>
                    <button onClick={handleRandomPredict} disabled={isLoading} className="w-full py-5 bg-teal-600 text-white rounded-2xl font-black shadow-lg">CHẠY NGẪU NHIÊN</button>
                  </div>
                  <ResultsList results={results} isLoading={isLoading} isMulti />
               </div>
            </div>
          )}

          {activeTab==='multi' && (
            <div className="flex-1 flex w-full h-full overflow-hidden">
               <div className="w-0 flex-1 bg-slate-100 dark:bg-black/20 relative">
                  <ForceGraph2D ref={fgMulti} graphData={graphData} nodeCanvasObject={paintNode} onNodeClick={setSelectedNode} onNodeHover={handleHover} backgroundColor={darkMode?'#0f172a':'#f8fafc'}/>
                  {selectedNode && <DetailDrawer node={selectedNode} onClose={()=>setSelectedNode(null)} />}
               </div>
               <div className="w-[380px] min-w-[380px] flex-shrink-0 flex flex-col bg-white dark:bg-slate-900 shadow-2xl z-30 border-l dark:border-slate-800 overflow-y-auto p-8 space-y-8">
                  <h3 className="text-xl font-black text-teal-600 flex items-center gap-2"><Layers size={24}/> Many-to-Many</h3>
                  <div className="space-y-6">
                    <div className="space-y-3">
                      <label className="text-[10px] font-black text-slate-400 uppercase">Thuốc (phẩy)</label>
                      <textarea rows="4" value={multiDrugsStr} onChange={e=>setMultiDrugsStr(e.target.value)} className="w-full p-4 bg-slate-50 dark:bg-slate-950 border dark:border-slate-800 rounded-2xl text-xs font-bold outline-none" placeholder="Aspirin, Ibuprofen..."/>
                    </div>
                    <div className="space-y-3">
                      <label className="text-[10px] font-black text-slate-400 uppercase">Bệnh (phẩy)</label>
                      <textarea rows="4" value={multiDiseasesStr} onChange={e=>setMultiDiseasesStr(e.target.value)} className="w-full p-4 bg-slate-50 dark:bg-slate-950 border dark:border-slate-800 rounded-2xl text-xs font-bold outline-none" placeholder="Diabetes..."/>
                    </div>
                    <button onClick={handleMultiPredict} disabled={isLoading} className="w-full py-5 bg-teal-600 text-white rounded-2xl font-black shadow-lg">PHÂN TÍCH CHÉO</button>
                  </div>
                  <ResultsList results={results} isLoading={isLoading} isMulti />
               </div>
            </div>
          )}

          {activeTab==='hyperparams' && (
            <div className="flex-1 overflow-y-auto p-8 grid grid-cols-1 lg:grid-cols-2 gap-8 animate-in slide-in-from-right">
               <div className="bg-white dark:bg-slate-900 p-8 rounded-[32px] border dark:border-slate-800 shadow-sm">
                  <h3 className="text-xl font-black mb-8 text-teal-600 flex items-center gap-3"><Settings size={22}/> Cấu hình Siêu tham số</h3>
                  <div className="grid grid-cols-2 gap-4">
                    {Object.entries(hyperData.params||{}).map(([k,v])=>(<div key={k} className="p-5 bg-slate-50 dark:bg-slate-800/50 border dark:border-slate-800 rounded-2xl"><p className="text-[10px] font-black text-slate-400 uppercase mb-1">{k}</p><p className="text-lg font-black text-teal-500">{v}</p></div>))}
                  </div>
               </div>
               <div className="bg-white dark:bg-slate-900 p-8 rounded-[32px] border dark:border-slate-800 shadow-sm">
                  <h3 className="text-xl font-black mb-8 text-rose-600 flex items-center gap-3"><CheckSquare size={22}/> Hiệu năng dự đoán</h3>
                  <div className="h-[400px]"><ResponsiveContainer><BarChart data={hyperData.metrics||[]} layout="vertical" margin={{left:20}}><XAxis type="number" hide domain={[0,1]}/><YAxis dataKey="name" type="category" width={80} tick={{fontSize:10,fontWeight:'bold'}}/><RechartsTooltip /><Bar dataKey="Improved" name="AMNTDDA" fill="#0d9488" radius={[0,8,8,0]}/></BarChart></ResponsiveContainer></div>
               </div>
            </div>
          )}

          {activeTab==='comparison' && (
            <div className="flex-1 p-8 overflow-y-auto text-center">
               <div className="bg-white dark:bg-slate-900 p-12 rounded-[48px] border dark:border-slate-800 shadow-sm max-w-5xl mx-auto">
                  <h3 className="text-3xl font-black mb-12 text-teal-600 uppercase tracking-tighter">So sánh Baseline vs AMNTDDA Improved</h3>
                  <div className="h-[500px]"><ResponsiveContainer><BarChart data={datasetName==='B-dataset'?performanceDataB:datasetName==='C-dataset'?performanceDataC:performanceDataF}><CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.1}/><XAxis dataKey="metric"/><YAxis domain={[0.6, 1.0]} /><RechartsTooltip contentStyle={{borderRadius:'24px'}}/><Legend /><Bar dataKey="Improved" name="AMNTDDA (Ours)" fill="#0d9488" radius={[12,12,0,0]}/><Bar dataKey="Original" name="Baseline Models" fill="#94a3b8" radius={[12,12,0,0]}/></BarChart></ResponsiveContainer></div>
               </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

function NavItem({ active, icon, label, onClick }) {
  return (
    <button onClick={onClick} className={cn("w-full flex items-center gap-4 px-5 py-4 rounded-[20px] transition-all font-bold", active?"bg-teal-600 text-white shadow-xl shadow-teal-600/30":"text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800")}>
      <span className={active?"text-white":"text-slate-400"}>{icon}</span><span className="text-sm truncate">{label}</span>
    </button>
  );
}

function ResultsList({ results, isLoading, isMulti = false }) {
  if (isLoading || results.length === 0) return null;
  return (
    <div className="space-y-3 pt-4 animate-in slide-in-from-bottom duration-500">
      <h4 className="text-[10px] font-black text-slate-400 uppercase border-b pb-2">Kết Quả Phân Tích</h4>
      {results.slice(0, 50).map((r, i) => (
        <div key={i} className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-2xl border dark:border-slate-800 flex justify-between items-center group hover:border-teal-500 transition-all">
          <div className="flex flex-col">
            <span className="text-[8px] font-black text-slate-400 uppercase tracking-widest">{isMulti ? `${r.source} ➔` : `RANK #${i+1}`}</span>
            <span className="font-bold text-slate-700 dark:text-slate-200 truncate w-40">{r.target}</span>
          </div>
          <span className="font-black text-teal-600">{(r.score*100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

function DetailDrawer({ node, onClose }) {
  const [imgIndex, setImgIndex] = useState(0);
  
  useEffect(() => {
    if (node.group === 'drug') {
      setImgIndex(0);
    }
  }, [node]);

  const getImgUrl = () => {
    const d_name = node.name.replace('Drug_','').trim();
    const urls = [];
    if (node.realId && node.realId.startsWith('DB')) {
       // Direct DrugBank SVG. Requires referrerPolicy="no-referrer" to bypass hotlink protection.
       urls.push(`https://go.drugbank.com/structures/small_molecule_drugs/${node.realId}.svg`);
    }
    // NCI Cactus (Very reliable for name-to-structure resolution)
    urls.push(`https://cactus.nci.nih.gov/chemical/structure/${encodeURIComponent(d_name)}/image`);
    
    // PubChem (Might be rate-limited if IP was banned)
    urls.push(`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${encodeURIComponent(d_name)}/PNG`);
    
    if (node.smiles) {
       urls.push(`https://www.simolecule.com/cdkdepict/depict/bow/svg?smi=${encodeURIComponent(node.smiles)}&w=300&h=300`);
    }
    
    urls.push(`https://placehold.co/300x300/e2e8f0/0d9488?text=${encodeURIComponent(d_name.substring(0, 12))}`);
    
    return urls[Math.min(imgIndex, urls.length - 1)];
  };

  return (
    <div className="absolute top-6 left-6 bottom-6 w-96 bg-white/95 dark:bg-slate-900/95 backdrop-blur-xl border border-teal-500/20 rounded-[40px] shadow-2xl z-50 flex flex-col p-8 animate-in slide-in-from-left duration-500">
      <div className="flex justify-between items-start mb-8">
        <div className="flex-1 truncate pr-4">
          <span className={cn("px-4 py-1 text-[8px] font-black rounded-full uppercase", node.group==='drug'?'bg-teal-100 text-teal-700':'bg-rose-100 text-rose-700')}>{node.group}</span>
          <h4 className="font-black text-2xl mt-3 dark:text-white truncate" title={node.name}>{node.name}</h4>
          <p className="text-[10px] font-bold text-slate-400 mt-2 uppercase tracking-widest">ID: <span className="text-teal-600">{node.realId||'N/A'}</span></p>
        </div>
        <button onClick={onClose} className="p-3 bg-slate-50 dark:bg-slate-800 hover:bg-rose-600 hover:text-white rounded-full transition-all text-slate-400 shadow-sm"><X size={20}/></button>
      </div>

      <div className="flex-1 overflow-y-auto space-y-8 pr-2 custom-scrollbar">
        {node.group === 'drug' ? (
          <div className="space-y-4">
             <div className="aspect-square bg-white rounded-[32px] border border-slate-100 p-6 shadow-inner flex items-center justify-center overflow-hidden group relative">
                <img 
                  key={`${node.id}-${imgIndex}`}
                  src={getImgUrl()} 
                  alt="Structure" 
                  referrerPolicy="no-referrer"
                  className="w-full h-full object-contain group-hover:scale-110 transition-transform duration-700" 
                  onError={() => { 
                    setImgIndex(prev => prev + 1);
                  }}
                />
             </div>
             <p className="text-[10px] text-center text-slate-400 font-bold italic uppercase tracking-widest">2D Molecular Structure</p>
          </div>
        ) : (
          <div className="aspect-square bg-rose-50/50 dark:bg-rose-900/10 rounded-[32px] border border-rose-100 dark:border-rose-900/30 flex flex-col items-center justify-center p-10 text-center">
             <Info size={48} className="text-rose-300 mb-4" />
             <p className="font-black text-rose-500 uppercase text-[10px] tracking-widest">Disease Entity</p>
             <p className="text-xs text-slate-400 mt-4 leading-relaxed">Thông tin cấu trúc hóa học không khả dụng cho các thực thể Bệnh lý.</p>
          </div>
        )}

        {node.smiles && (
          <div className="space-y-3">
            <span className="text-[10px] font-black text-slate-400 uppercase flex items-center gap-2 tracking-widest"><Layers size={14} className="text-teal-500"/> SMILES Notation</span>
            <div className="p-6 bg-slate-50 dark:bg-slate-800/50 rounded-3xl border dark:border-slate-800 group relative">
              <p className="text-[11px] font-mono break-all text-slate-600 dark:text-slate-300 font-bold leading-relaxed">{node.smiles}</p>
              <button className="absolute top-2 right-2 p-2 opacity-0 group-hover:opacity-100 transition-opacity bg-white dark:bg-slate-700 rounded-lg shadow-sm border dark:border-slate-600">
                <Clipboard size={14} className="text-teal-600"/>
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="pt-8 border-t dark:border-slate-800 mt-4">
        <button onClick={onClose} className="w-full py-4 bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900 rounded-2xl font-black text-[10px] uppercase tracking-widest transition-all active:scale-95">ĐÓNG CHI TIẾT</button>
      </div>
    </div>
  );
}
