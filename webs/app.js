'use strict';
(() => {
  const root = document.getElementById('cnn-lab-preview-v2');
  const $ = id => document.getElementById(id);
  ['cl-start','cl-evaluate','cl-confirm'].forEach(id => $(id).disabled = true);
  const labels = {cat:'貓', dog:'狗', fox:'狐狸', leopard:'豹', lion:'獅子', wolf:'狼'};
  let reviewed = null, selectionDriven = false;
  let state = null, runId = null, metric = 'loss', epoch = 10, batch = 32;
  let matrixUnit = 'count', chosenCell = null, watchedJob = null, polling = false, runSnapshot = '';
  const shownSeries = {train:true, validation:true};
  const percent = value => Number.isFinite(value) ? (value * 100).toFixed(2) + '%' : '—';
  const decimal = value => Number.isFinite(value) ? value.toFixed(4) : '—';
  const duration = seconds => Number.isFinite(seconds) ? `${Math.floor(seconds / 60)} 分 ${Math.floor(seconds % 60)} 秒` : '尚無完整耗時';
  const run = () => {
    const item=state?.runs.find(item => item.id === runId);
    if(!item)return null;
    const history=item.history.filter(row=>row.epoch<=epoch);
    const best=history.reduce((a,b)=>!a||b[item.accuracy_key]>a[item.accuracy_key]?b:a,null);
    return {...item,history,best,training:{...item.training,epochs:epoch},
      actual_epochs:item.training.epochs,
      status:history.length>=epoch?'completed':item.status==='completed'?'incomplete':item.status,
      elapsed_seconds:item.milestones?.[epoch]?.elapsed_seconds ?? (epoch===item.training.epochs?item.elapsed_seconds:undefined),
      evaluation:epoch===10?item.evaluation:undefined,
      can_evaluate:epoch===10&&item.can_evaluate};
  };
  const active = () => state?.jobs.find(job => ['queued','running','cancel_requested'].includes(job.status));
  const displayName = name => labels[name] || name;

  function alert(message) { $('cl-alert').textContent = message || ''; $('cl-alert').hidden = !message; }
  async function api(path, data) {
    const options = data === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json','X-Lab-Token':state.token}, body:JSON.stringify(data)};
    const response = await fetch(path, options);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '操作失敗');
    return result;
  }
  function params() { return {model:$('cl-model').value, epochs:10, batch_size:batch, learning_rate:$('cl-lr').value === 'all' ? 'all' : Number($('cl-lr').value), optimizer:$('cl-optimizer').value}; }
  function draft() { $('cl-batches').textContent = `每輪 ${Math.ceil(19626 / batch).toLocaleString()} 批 · 訓練 10 輪 · 查看前 ${epoch} 輪`; $('cl-draft-status').textContent = '切換參數會尋找相符紀錄；輪數切換只顯示該節點以前的結果，新訓練固定 10 輪。'; }
  function selectTab(tab) {
    const buttons = [...root.querySelectorAll('[role=tab]')];
    buttons.forEach(button => { button.setAttribute('aria-selected', String(button === tab)); $(button.getAttribute('aria-controls')).hidden = button !== tab; });
    $('cl-original-stats').hidden = tab.id === 'cl-tab-class';
    root.querySelector('.lab-result-head .lab-eyebrow').textContent = '本機實測紀錄 · 驗證集結果';
    root.querySelector('.lab-result-head h2').textContent = tab.id === 'cl-tab-class' ? '看懂每一類的判斷' : '看見模型如何學習';
    requestAnimationFrame(drawChart);
  }
  function refreshRunPicker() {
    const picker = $('cl-run');
    const wanted = runId;
    picker.replaceChildren();
    for (const item of state.runs) {
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = `${state.models[item.model] || item.model} · ${item.training.epochs} 輪 · LR ${item.training.learning_rate} · ${item.training.optimizer} · ${item.status} · ${item.id.slice(-6)}`;
      option.title = item.name;
      picker.append(option);
    }
    if(selectionDriven && !wanted){runId=null;picker.selectedIndex=-1;return;}
    if (state.runs.some(item => item.id === wanted)) runId = wanted;
    else runId = state.jobs.find(job => job.status === 'running')?.run_id || state.runs.filter(item => item.status === 'completed').sort((a,b) => (b.best?.[b.accuracy_key] || 0) - (a.best?.[a.accuracy_key] || 0))[0]?.id || state.runs.at(-1)?.id || null;
    if (runId) picker.value = runId;
    if(!wanted && runId){const item=state.runs.find(r=>r.id===runId);$('cl-model').value=item.model;$('cl-lr').value=String(item.training.learning_rate);$('cl-optimizer').value=item.training.optimizer;}
  }
  function renderJob() {
    const current = state.jobs.find(job => job.status === 'running') || active() || state.jobs.find(job => job.id === watchedJob);
    $('cl-job').hidden = !current;
    if (!current) return;
    const group = current.group_id ? state.jobs.filter(job => job.group_id === current.group_id) : [current];
    const finished = group.filter(job => !['queued','running','cancel_requested'].includes(job.status)).length;
    const names = {queued:'等待依序執行',preparing:'準備模型與資料',training:'訓練中',validation:'驗證中',loading_model:'載入最佳權重',completed:'工作完成',cancelled:'已取消',failed:'工作失敗',interrupted:'工作中斷'};
    $('cl-job-title').textContent = `${current.kind === 'train' ? `模型訓練 · 排程 ${finished}/${group.length}` : '驗證集評估'} · ${names[current.phase || current.status] || '準備中'}`;
    const elapsed = current.status === 'running' ? Date.now()/1000 - current.started_at : current.elapsed_seconds;
    $('cl-job-detail').textContent = `${current.kind === 'train' ? `Epoch ${current.epoch || 0} / ${current.epochs} · ` : ''}${current.batch ? `第 ${current.batch} 批 · ` : ''}${duration(elapsed)} · ${current.execution?.label || 'Mac CPU'}`;
    const progress = $('cl-progress');
    if (current.status === 'completed') { progress.value = 100; }
    else if (current.kind === 'validation' && current.batches) progress.value = 100 * current.batch / current.batches;
    else if (current.training_completed) progress.value = 99;
    else if (current.kind === 'train' && current.epochs) progress.value = Math.min(99, 100 * ((current.epochs_completed || 0) + (current.phase === 'training' ? Math.min(.99, (current.batch || 0) / (current.batches || 1)) : current.phase === 'validation' ? 1 : 0)) / current.epochs);
    else progress.removeAttribute('value');
    $('cl-cancel').hidden = !['queued','running','cancel_requested'].includes(current.status);
    $('cl-cancel').disabled = !!current.cancel_requested;
    $('cl-job-message').textContent = current.message || (current.cancel_requested ? '已要求取消，等待目前這批停止。' : '紀錄與權重會分開保存，不覆寫先前實驗。');
  }
  function renderResult() {
    const item = run(), busy = !!active();
    $('cl-start').disabled = busy;
    $('cl-confirm').disabled = busy;
    $('cl-evaluate').disabled = busy || !item?.can_evaluate;
    $('cl-best').textContent = item?.best ? `最佳 · 第 ${item.best.epoch} 輪` : '尚無完整輪次';
    $('cl-accuracy').textContent = percent(item?.best?.[item?.accuracy_key]);
    $('cl-loss').textContent = decimal(item?.evaluation?.loss ?? item?.best?.val_loss);
    const oldStats = $('cl-original-stats').querySelectorAll('.lab-stat');
    oldStats[2].querySelector('.lab-value').textContent = percent(item?.evaluation?.report?.['macro avg']?.['f1-score']);
    oldStats[2].querySelector('.lab-stat-foot').textContent = item?.evaluation ? '最佳權重 · 驗證集' : epoch<10?'分類報告僅顯示於 10 輪節點':'尚未計算驗證分類指標';
    $('cl-complete').textContent = item ? `${item.status === 'completed' ? '已完成' : '目前'} ${item.history.length} / ${item.training.epochs} 輪 · ${item.status === 'completed' ? '最佳權重已保存' : '尚未完成實驗'}` : '尚無實驗';
    $('cl-duration').textContent = `本機 · ${duration(item?.elapsed_seconds)}`;
    $('cl-result-meta').textContent = item ? `${state.models[item.model]} · ${item.training.optimizer} · Batch ${item.training.batch_size} · LR ${item.training.learning_rate} · ${item.execution?.label || 'Mac CPU'}` : '尚未選擇實驗';
    $('cl-chart-label').textContent = `${metric === 'loss' ? 'Loss' : 'Accuracy'} 學習曲線`;
    $('cl-curve-note').textContent = (item?.history.length || 0) < 2 ? '資料不足兩輪，不能判定欠擬合或過擬合。' : '請觀察訓練與驗證的多輪趨勢，不以單輪波動判定過擬合。';
    renderClassification();renderComparison();renderJob();drawChart();
  }
  function renderComparison() {
    const body = $('cl-compare-body');body.replaceChildren();
    const comparisons = state.runs.filter(item =>
      item.status === 'completed' && item.training.epochs === 10 && item.history.length === 10 &&
      ($('cl-filter-model').value === 'all' || item.model === $('cl-filter-model').value) &&
      ($('cl-filter-opt').value === 'all' || item.training.optimizer === $('cl-filter-opt').value) &&
      ($('cl-filter-lr').value === 'all' || item.training.learning_rate === Number($('cl-filter-lr').value))
    );
    for (const item of comparisons) {
      const tr = document.createElement('tr');
      const values = [state.models[item.model] || item.model, `${item.history.length}/${item.training.epochs}`, item.training.batch_size, item.training.learning_rate, item.training.optimizer, percent(item.best?.[item.accuracy_key]), decimal(item.evaluation?.loss ?? item.best?.val_loss), percent(item.evaluation?.report?.['macro avg']?.['f1-score']), duration(item.elapsed_seconds), item.execution?.label || 'Mac CPU', item.status];
      values.forEach(value => {const td=document.createElement('td');td.textContent=value;tr.append(td);});
      tr.title = item.message || `基準版本 ${item.protocol_id || '舊實驗'}`;
      body.append(tr);
    }
  }
  function renderClassification() {
    const item = run(), result = item?.evaluation;
    const names = result ? Object.entries(result.class_indices).sort((a,b)=>a[1]-b[1]).map(x=>x[0]) : Object.keys(state.dataset.val || {}).sort();
    const report = result?.report;
    const support = names.map(name => report?.[name]?.support ?? state.dataset.val?.[name] ?? 0);
    $('cl-class-provenance').textContent = result ? `驗證集實測 · ${new Date(result.evaluated_at*1000).toLocaleString()}` : '驗證集分類指標尚未計算';
    $('cl-class-banner').textContent = result ? '此處為所選實驗最佳權重的實際驗證結果。百分比按真實類別的每列張數正規化。' : '每組訓練完成後，會使用該組最佳權重計算驗證指標；空值不代表 0。';
    $('cl-class-accuracy').textContent = percent(result?.accuracy);
    $('cl-class-macro').textContent = percent(report?.['macro avg']?.['f1-score']);
    $('cl-class-weighted').textContent = percent(report?.['weighted avg']?.['f1-score']);
    $('cl-score-basis').textContent = `驗證集 · ${support.reduce((a,b)=>a+b,0).toLocaleString()} 張`;
    $('cl-matrix-caption').textContent = result ? '驗證實測 · 點格子查看分類方向' : '尚未計算，空值不代表 0';
    const body=$('cl-matrix-body');body.replaceChildren();
    names.forEach((name,i)=>{const tr=document.createElement('tr'),th=document.createElement('th');th.scope='row';th.textContent=displayName(name);tr.append(th);names.forEach((pred,j)=>{const td=document.createElement('td'),button=document.createElement('button'),value=result?.confusion_matrix[i][j],ratio=support[i] ? value/support[i] : 0;button.type='button';button.dataset.row=i;button.dataset.col=j;button.textContent=result ? matrixUnit==='count' ? value : (ratio*100).toFixed(1)+'%' : '—';button.disabled=!result;button.setAttribute('aria-label',`真實${displayName(name)}，預測${displayName(pred)}，${result ? value+' 張' : '尚未計算'}`);button.setAttribute('aria-pressed',String(!!chosenCell && chosenCell[0]===i && chosenCell[1]===j));if(result){button.style.background=`color-mix(in srgb,var(--lab-blue) ${8+ratio*72}%,var(--lab-panel))`;if(ratio>.6)button.style.color='light-dark(#fff,#171719)';}button.addEventListener('click',()=>{chosenCell=[i,j];renderClassification();});td.append(button);tr.append(td);});body.append(tr);});
    const table=$('cl-scores-body');table.replaceChildren();
    for(const name of names){const tr=document.createElement('tr'),row=report?.[name];[displayName(name),percent(row?.precision),percent(row?.recall),percent(row?.['f1-score']),row?.support ?? state.dataset.val?.[name] ?? 0].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});table.append(tr);}
    const foot=$('cl-scores-foot');foot.replaceChildren();for(const key of ['macro avg','weighted avg']){const tr=document.createElement('tr'),row=report?.[key];[key==='macro avg'?'Macro avg':'Weighted avg',percent(row?.precision),percent(row?.recall),percent(row?.['f1-score']),row?.support ?? support.reduce((a,b)=>a+b,0)].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});foot.append(tr);}
    $('cl-errors').hidden=true;$('cl-errors').replaceChildren();$('cl-image-note').textContent='';
    if(result && chosenCell){const [i,j]=chosenCell,value=result.confusion_matrix[i][j];$('cl-cell-title').textContent=`${displayName(names[i])} → ${displayName(names[j])} · ${i===j?'正確分類':'誤判'}`;$('cl-cell-detail').textContent=`${value} 張，占真正${displayName(names[i])}的 ${percent(support[i]?value/support[i]:0)}。`;$('cl-error-images').disabled=i===j||value===0;}else{$('cl-cell-title').textContent='選擇一格，查看模型的判斷';$('cl-cell-detail').textContent=result?'非對角線格子可查看被認錯的照片。':'尚未載入實際預測結果。';$('cl-error-images').disabled=true;}
    $('cl-eval-note').textContent = result ? `已核對設定與類別順序 · Loss ${decimal(result.loss)} · Top3 ${percent(result.top3_accuracy)}` : '評估只使用驗證集，不讀取測試集。';
  }
  function drawChart() {
    if($('cl-panel-curves').hidden)return;
    const container=root.querySelector('[data-viz-container]'),item=run();
    const rows=item?.history || [];container.replaceChildren();
    if(!rows.length){const empty=document.createElement('p');empty.className='lab-help';empty.textContent='尚無完整輪次，訓練完成第一輪後會顯示曲線。';container.append(empty);return;}
    const width=Math.max(200,Math.floor(container.getBoundingClientRect().width)),height=280,L=48,R=18,T=18,B=50,W=width-L-R,H=height-T-B;
    const trainKey=metric==='accuracy'?(item.accuracy_key==='val_main_accuracy'?'main_accuracy':'accuracy'):'loss';
    const valKey=metric==='accuracy'?item.accuracy_key:'val_loss';
    const values=rows.flatMap(row=>[row[trainKey],row[valKey]]).filter(Number.isFinite);
    const max=metric==='accuracy'?1:Math.max(.1,...values)*1.08,last=rows.at(-1).epoch;
    const x=value=>L+(last===1?W/2:(value-1)/(last-1)*W),y=value=>T+H*(1-value/max),ns='http://www.w3.org/2000/svg';
    const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.setAttribute('role','img');svg.setAttribute('aria-label',`${metric} 訓練與驗證曲線，${rows.length} 輪`);
    const add=(tag,attrs,text)=>{const el=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([key,value])=>el.setAttribute(key,value));if(text!==undefined)el.textContent=text;svg.append(el);return el;};
    add('title',{},'實際訓練與驗證曲線');
    for(let i=0;i<=4;i++){const value=max*i/4;add('line',{x1:L,x2:width-R,y1:y(value),y2:y(value),class:'lab-chart-grid'});add('text',{x:L-7,y:y(value)+4,'text-anchor':'end',class:'lab-chart-text'},metric==='accuracy'?Math.round(value*100)+'%':Number(value.toFixed(2)));}
    rows.forEach((row,i)=>{if(i===0||i===rows.length-1||(width>400?i%2===0:i===Math.floor(rows.length/2)))add('text',{x:x(row.epoch),y:height-27,'text-anchor':'middle',class:'lab-chart-text'},row.epoch);});add('text',{x:L+W/2,y:height-6,'text-anchor':'middle',class:'lab-chart-text'},'Epoch');
    const readout=document.createElement('p');readout.className='lab-plot-readout';readout.textContent='移到圖上查看每輪數值';
    const format=metric==='accuracy'?percent:decimal;
    for(const [key,name,color,series] of [[trainKey,'訓練','var(--lab-blue)','train'],[valKey,'驗證','var(--lab-orange)','validation']]){if(!shownSeries[series])continue;const valid=rows.filter(row=>Number.isFinite(row[key]));if(valid.length>1)add('polyline',{points:valid.map(row=>`${x(row.epoch)},${y(row[key])}`).join(' '),fill:'none',stroke:color,'stroke-width':2.5,'stroke-dasharray':series==='validation'?'6 3':'none'});valid.forEach(row=>add('circle',{cx:x(row.epoch),cy:y(row[key]),r:4,fill:color}));}
    svg.addEventListener('pointermove',event=>{const px=(event.clientX-svg.getBoundingClientRect().left)*width/svg.getBoundingClientRect().width;const selected=rows.reduce((a,b)=>Math.abs(x(a.epoch)-px)<Math.abs(x(b.epoch)-px)?a:b);readout.textContent=`Epoch ${selected.epoch} · 訓練 ${format(selected[trainKey])} · 驗證 ${format(selected[valKey])}`;});
    container.append(svg,readout);const legend=document.createElement('div');legend.className='lab-local-legend';
    for(const [key,label,color] of [['train','訓練 · 實線','var(--lab-blue)'],['validation','驗證 · 虛線','var(--lab-orange)']]){const button=document.createElement('button'),mark=document.createElement('span');button.type='button';button.className='lab-legend-button';button.setAttribute('aria-pressed',String(shownSeries[key]));mark.style.background=color;mark.style.display='inline-block';mark.style.width='16px';mark.style.height='3px';mark.style.marginRight='6px';button.append(mark,document.createTextNode(label));button.addEventListener('click',()=>{shownSeries[key]=!shownSeries[key];drawChart();});legend.append(button);}container.append(legend);
    const details=document.createElement('details');details.className='lab-explain';const summary=document.createElement('summary');summary.textContent='每輪精確數值';details.append(summary);const table=document.createElement('table'),head=document.createElement('tr');['Epoch','訓練','驗證'].forEach(text=>{const th=document.createElement('th');th.textContent=text;head.append(th);});table.append(head);rows.forEach(row=>{const tr=document.createElement('tr');[row.epoch,format(row[trainKey]),format(row[valKey])].forEach(text=>{const td=document.createElement('td');td.textContent=text;tr.append(td);});table.append(tr);});details.append(table);container.append(details);
  }
  async function poll() {
    if(polling)return;polling=true;
    try { const next=await api('/api/state');if($('cl-alert').textContent.startsWith('無法連線到本機介面：'))alert('');const first=!state;const snapshot=JSON.stringify(next.runs.map(item=>({...item,elapsed_seconds:undefined})));state=next;if(first){$('cl-model').replaceChildren();Object.entries(state.models).forEach(([value,label])=>{const option=document.createElement('option');option.value=value;option.textContent=label;$('cl-model').append(option);});Object.entries(state.models).forEach(([value,label])=>{const option=document.createElement('option');option.value=value;option.textContent=label;$('cl-filter-model').append(option);});$('cl-model').value='efficientnet_b0';configureMode();}if(first||snapshot!==runSnapshot){runSnapshot=snapshot;refreshRunPicker();renderResult();}else{const busy=!!active();$('cl-start').disabled=busy;$('cl-confirm').disabled=busy;$('cl-evaluate').disabled=busy||!run()?.can_evaluate;renderJob();} }
    catch(error){alert('無法連線到本機介面：'+error.message);}finally{polling=false;}
  }
  root.querySelectorAll('[data-choice]').forEach(group=>group.addEventListener('click',event=>{const button=event.target.closest('button');if(!button)return;group.querySelectorAll('button').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));if(group.dataset.choice==='epoch')epoch=Number(button.dataset.value);else if(group.dataset.choice==='batch')batch=Number(button.dataset.value);else metric=button.dataset.value;draft();if(group.dataset.choice==='epoch')selectMatchingRun();if(group.dataset.choice==='metric')renderResult();else drawChart();}));
  for(const id of ['cl-model','cl-lr','cl-optimizer'])$(id).addEventListener('change',()=>{draft();selectMatchingRun();});
  const optimizers=[['adam','Adam'],['sgd','SGD'],['adamw','AdamW'],['rmsprop','RMSprop']];$('cl-optimizer').replaceChildren();optimizers.forEach(([value,label])=>{const option=document.createElement('option');option.value=value;option.textContent=label;$('cl-optimizer').append(option);});


  function selectMatchingRun() {
    if(!state)return;
    selectionDriven=true;
    const p=params();
    const matches=state.runs.filter(item=>item.model===p.model && item.training.optimizer===p.optimizer && item.training.learning_rate===p.learning_rate && item.training.batch_size===32 && item.training.epochs>=epoch);
    matches.sort((a,b)=>(b.training.epochs===10)-(a.training.epochs===10) || b.history.length-a.history.length);
    runId=matches[0]?.id || null;
    if(runId)$('cl-run').value=runId;
    else $('cl-run').selectedIndex=-1;
    chosenCell=null;renderResult();
  }

  function configureMode() {
    const mode=$('cl-mode').value, multiple=mode!=='free';
    for(const id of ['cl-model','cl-lr','cl-optimizer']){
      const select=$(id);select.querySelector('option[value="all"]')?.remove();
      if(multiple){const option=document.createElement('option');option.value='all';option.textContent='全部';select.prepend(option);}
      if(mode==='grid')select.value='all';
    }
    if(mode!=='free')epoch=10;
    root.querySelectorAll('[data-choice="epoch"] button').forEach(button=>{button.disabled=mode!=='free';button.setAttribute('aria-pressed',String(Number(button.dataset.value)===epoch));});
    reviewed=null;$('cl-review').hidden=true;draft();
    $('cl-draft-status').textContent=mode==='grid'?'全部模型 × 3 種學習率 × 4 種優化器，共 120 組；每組訓練 10 輪，保存 1／5／10 節點。':mode==='comparison'?'固定 10 輪，符合相同基準的已完成實驗直接沿用。':'沒有權重也能從零訓練；需先準備環境與分類圖片。';
  }
  $('cl-mode').addEventListener('change',configureMode);
  for(const id of ['cl-filter-model','cl-filter-opt','cl-filter-lr'])$(id).addEventListener('change',renderComparison);

  const tabs=[...root.querySelectorAll('[role=tab]')];tabs.forEach((tab,index)=>{tab.addEventListener('click',()=>selectTab(tab));tab.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;selectTab(tabs[next]);tabs[next].focus();});});
  $('cl-run').addEventListener('change',()=>{runId=$('cl-run').value;const item=state.runs.find(r=>r.id===runId);if(item){$('cl-model').value=item.model;$('cl-lr').value=String(item.training.learning_rate);$('cl-optimizer').value=item.training.optimizer;}chosenCell=null;renderResult();});
  $('cl-start').addEventListener('click',()=>{const p=params();reviewed={path:$('cl-mode').value==='grid'?'/api/grid':$('cl-mode').value==='comparison'?'/api/comparison':'/api/train',values:p};$('cl-review-values').textContent=`${state.models[p.model] || '全部模型'} · 10 Epoch（保存 1／5／10 節點） · Batch ${p.batch_size} · LR ${p.learning_rate} · ${p.optimizer} · ${$('cl-mode').value==='grid' ? (p.model==='all'?10:1)*(p.learning_rate==='all'?3:1)*(p.optimizer==='all'?4:1) : (p.model==='all'?10:1)*(p.learning_rate==='all'?3:1)*(p.optimizer==='all'?4:1)} 組`;$('cl-review').hidden=false;$('cl-review').scrollIntoView({block:'nearest',behavior:'smooth'});});
  $('cl-close-review').addEventListener('click',()=>{$('cl-review').hidden=true;$('cl-start').focus();});
  $('cl-confirm').addEventListener('click',async()=>{try{$('cl-confirm').disabled=true;alert('');if(!reviewed)return;const job=await api(reviewed.path,reviewed.values);watchedJob=job.id;if(job.id)runId=job.id;$('cl-review').hidden=true;alert(`已排入 ${job.count} 組；沿用 ${job.reused?.length || 0} 組既有結果。`);await poll();}catch(error){alert(error.message);}finally{$('cl-confirm').disabled=!!active();}});
  $('cl-evaluate').addEventListener('click',async()=>{try{$('cl-evaluate').disabled=true;alert('');const job=await api('/api/validate',{run_id:runId});watchedJob=job.id;await poll();}catch(error){alert(error.message);$('cl-evaluate').disabled=!!active();}});
  $('cl-cancel').addEventListener('click',async()=>{try{const job=active();if(!job)return;await api('/api/cancel',{id:job.id});$('cl-cancel').disabled=true;$('cl-job-message').textContent='已要求取消，等待目前這批停止。';}catch(error){alert(error.message);}});
  $('cl-matrix-units').addEventListener('click',event=>{const button=event.target.closest('button');if(!button)return;matrixUnit=button.dataset.unit;$('cl-matrix-units').querySelectorAll('button').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));renderClassification();});
  $('cl-error-images').addEventListener('click',async()=>{try{const [truth,pred]=chosenCell;const rows=await api(`/api/errors?run=${encodeURIComponent(runId)}&true=${truth}&pred=${pred}`);const grid=$('cl-errors');grid.replaceChildren();grid.hidden=false;rows.forEach(row=>{const figure=document.createElement('figure'),image=document.createElement('img'),caption=document.createElement('figcaption');image.src=`/api/image?run=${encodeURIComponent(runId)}&index=${row.index}`;image.alt=`真實與預測不一致的驗證照片：${row.filename}`;image.loading='lazy';caption.textContent=`${row.filename} · 預測分數 ${percent(row.confidence)}`;figure.append(image,caption);grid.append(figure);});$('cl-image-note').textContent=`顯示此格前 ${rows.length} 張；預測分數不是校準後的正確機率。`;grid.scrollIntoView({block:'nearest',behavior:'smooth'});}catch(error){alert(error.message);}});
  // 比較表可同時容納不同模型與新增的分類指標。
  const header=$('cl-panel-compare').querySelector('thead tr');header.replaceChildren();['模型','輪數','Batch','LR','Optimizer','驗證 Acc.','驗證 Loss','Macro F1','耗時','執行環境','狀態'].forEach(text=>{const th=document.createElement('th');th.textContent=text;header.append(th);});
  selectTab($('cl-tab-curves'));draft();poll();setInterval(poll,3000);
  if(globalThis.ResizeObserver)new ResizeObserver(()=>requestAnimationFrame(drawChart)).observe(root.querySelector('[data-viz-container]'));
})();
