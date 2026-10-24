<template>
  <h1>资管中台</h1>
  <div class="sub" v-if="isPE">项目管线 → 尽调 Agent（招股说明书）→ 投决 → 出资 → 交割 → 季度估值（IPEV / AAP）→ 上市与解禁 → 减持 → 分配瀑布 → LP 季报</div>
  <div class="sub" v-else-if="!isQuant">托管估值表 → 净值与估值核对 → 每日风控与合同限额 → 业绩归因 → 股票池与投委会 → 交易前检查 → 投资者月报</div>
  <div class="sub" v-else>因子库与因子检验 → 风险模型 → 约束优化 → 分日执行（成交量参与率、冲击成本）→ 托管估值表 → 风控合规 → 超额收益分解</div>
  <div class="card row">
    <select v-model="code" @change="tab = '产品概览'; loadAll()"><option v-for="p in products" :key="p.code" :value="p.code">{{ p.name }}</option></select>
    <span v-for="t in (isPE ? [] : tabs)" :key="t" class="chip" :style="tab === t ? 'background:var(--accent);color:#fff' : ''" @click="tab = t">{{ t }}</span>
  </div>

  <PEView v-if="isPE" :key="code" :code="code" :initial-tab="initialTab" />
  <template v-else-if="ov">
    <!-- 产品概览 -->
    <template v-if="tab === '产品概览'">
      <div class="grid g5" style="margin-top:14px">
        <div class="kpi"><span>单位净值</span><b>{{ m.unit_nav.toFixed(4) }}</b><span>{{ m.end }}</span></div>
        <div class="kpi"><span>成立以来 / 年化</span><b>{{ pct(m.total_return) }}</b><span>年化 {{ pct(m.ann_return) }}</span></div>
        <div class="kpi"><span>年化超额（{{ bname }}）</span><b :class="m.ann_excess >= 0 ? 'ok' : 'bad'">{{ pct(m.ann_excess) }}</b><span>信息比率 {{ m.info_ratio.toFixed(2) }}</span></div>
        <div class="kpi"><span>最大回撤</span><b>{{ pct(m.max_drawdown) }}</b><span>夏普 {{ m.sharpe.toFixed(2) }} · 卡玛 {{ m.calmar.toFixed(2) }}</span></div>
        <div class="kpi"><span>规模 / 待处理风控事件</span><b>{{ (ov.nav.at(-1).net_assets / 1e8).toFixed(2) }} 亿</b><span :class="ov.open_events ? 'bad' : 'ok'">{{ ov.open_events }} 条待处理</span></div>
      </div>
      <div class="card"><h3>单位净值 vs {{ bname }}（归一）</h3>
        <LineChart :labels="ov.nav.map((r) => r.date)" :base="1" :fmt="(v) => v.toFixed(2)"
          :series="[{ name: '单位净值', values: ov.nav.map((r) => r.unit_nav), color: '#0E7C66' }, { name: bname, values: ov.nav.map((r) => r.bench), color: '#aaa', width: 1.2 }]" /></div>
      <div class="grid g2">
        <div class="card"><h3>合同要素</h3>
          <table><tr><td>管理人 / 托管人</td><td>{{ ov.product.manager }} / {{ ov.product.custodian }}</td></tr>
            <tr><td>成立日 / 基准</td><td>{{ ov.product.inception }} / {{ bname }}</td></tr>
            <tr><td>管理费 / 托管费 / 业绩报酬</td><td>{{ pct(ov.product.fees.management) }} / {{ pct(ov.product.fees.custody, 2) }} / {{ pct(ov.product.fees.performance, 0) }}</td></tr>
            <template v-if="!isQuant">
            <tr><td>预警线 / 止损线</td><td>{{ ov.product.contract_limits.warning_line }} / {{ ov.product.contract_limits.stop_line }}</td></tr>
            <tr><td>单票 / 单行业 / 现金</td><td>≤ {{ pct(ov.product.contract_limits.single_name_max, 0) }} / ≤ {{ pct(ov.product.contract_limits.industry_max, 0) }} / ≥ {{ pct(ov.product.contract_limits.cash_min, 0) }}</td></tr></template>
            <template v-else>
            <tr><td>单票 / 行业偏离基准</td><td>≤ {{ pct(ov.product.contract_limits.active_single_max, 0) }} / ≤ {{ pct(ov.product.contract_limits.active_industry_max, 0) }}</td></tr>
            <tr><td>成分股占比 / 跟踪误差目标</td><td>≥ {{ pct(ov.product.contract_limits.bench_constituent_min, 0) }} / ≤ {{ pct(ov.product.contract_limits.te_max, 0) }}</td></tr></template></table></div>
        <div class="card"><h3>申购赎回（登记机构）</h3>
          <table><tr><th>日期</th><th>类型</th><th>金额</th><th>确认净值</th><th>份额</th></tr>
            <tr v-for="f in ov.flows" :key="f.date"><td>{{ f.date }}</td><td>{{ f.type === 'subscribe' ? '申购' : '赎回' }}</td><td>{{ wan(f.amount) }}</td><td>{{ f.nav.toFixed(4) }}</td><td>{{ (f.shares / 1e4).toFixed(1) }} 万份</td></tr></table></div>
      </div>
      <div class="card"><h3>月度收益</h3>
        <table><tr><th>月份</th><th>产品</th><th>{{ bname }}</th><th>超额</th></tr>
          <tr v-for="r in ov.monthly.slice(-12).reverse()" :key="r.month"><td>{{ r.month }}</td><td :class="r['组合'] < 0 ? 'neg' : ''">{{ pct(r['组合']) }}</td><td>{{ pct(r['基准']) }}</td><td :class="r['超额'] < 0 ? 'neg' : 'ok'">{{ pct(r['超额']) }}</td></tr></table></div>
    </template>

    <!-- 风控合规 -->
    <template v-if="tab === '风控合规'">
      <div class="grid g5" style="margin-top:14px">
        <div class="kpi"><span>1 日 95% VaR</span><b>{{ pct(r.var95_1d) }}</b><span>CVaR {{ pct(r.cvar95_1d) }}</span></div>
        <div class="kpi"><span>年化波动 / Beta</span><b>{{ pct(r.vol_ann, 1) }}</b><span>Beta {{ r.beta?.toFixed(2) }}</span></div>
        <div class="kpi"><span>股票仓位 / 现金</span><b>{{ pct(r.stock_weight, 1) }}</b><span>现金 {{ pct(r.cash_weight, 1) }}</span></div>
        <div class="kpi"><span>流动性（5 日可变现）</span><b>{{ pct(r.liquidity?.['5'] ?? r.liquidity?.[5], 0) }}</b><span>1 日 {{ pct(r.liquidity?.['1'] ?? r.liquidity?.[1], 0) }}</span></div>
        <div class="kpi" v-if="!isQuant"><span>距预警线 / 止损线</span><b>{{ pct(r.to_warning_line, 1) }}</b><span>止损线 {{ pct(r.to_stop_line, 1) }}</span></div>
        <div class="kpi" v-else><span>跟踪误差（60 日）</span><b>{{ pct(r.te_60d, 1) }}</b><span>成分股占比 {{ pct(r.bench_member_weight, 0) }}</span></div>
      </div>
      <div class="card"><h3>合同限额（{{ r.date }}）</h3>
        <table><tr><th>限额</th><th>实际</th><th>限值</th><th>说明</th><th>状态</th></tr>
          <tr v-for="l in r.limits" :key="l.rule"><td>{{ l.label }}</td><td>{{ l.rule.includes('line') ? l.value.toFixed(4) : (l.rule === 'banned' ? l.value : pct(l.value, 1)) }}</td>
            <td>{{ l.direction === 'min' ? '≥' : '≤' }} {{ l.rule.includes('line') || l.rule === 'banned' ? l.limit : pct(l.limit, 0) }}</td><td>{{ l.detail }}</td><td><span :class="['tagx', l.status]">{{ l.status }}</span></td></tr></table></div>
      <div class="grid g2">
        <div class="card"><h3>行业分布（申万一级，占净值）</h3>
          <div v-for="(v, k) in r.by_industry" :key="k" class="row" style="gap:6px;margin:2px 0"><span style="width:80px">{{ k }}</span>
            <div :style="{ width: v * 600 + 'px', height: '9px', background: v > ov.product.contract_limits.industry_max * 0.9 ? 'var(--amber)' : 'var(--accent)', borderRadius: '3px' }"></div><span class="muted">{{ pct(v, 1) }}</span></div></div>
        <div class="card"><h3>待处理风控事件（{{ events.length }}）</h3>
          <p class="muted" style="margin:0 0 6px">同一规则在相邻交易日连续触发合并为一条，按区间处理留痕</p>
          <div class="timeline" style="max-height:300px"><div v-for="e in events" :key="e.ids[0]">
            <span class="t" style="width:170px">{{ e.start === e.end ? e.start : e.start + ' ~ ' + e.end }}</span><span :class="['tagx', e.status]">{{ e.status }}</span>
            {{ e.detail }}（{{ e.days }} 天，极值 {{ e.rule.includes('line') ? e.peak.toFixed(4) : pct(e.peak, 1) }}）
            <span class="chip" @click="handle(e)">确认处理</span></div></div></div>
      </div>
    </template>


    <!-- 量化：超额归因 -->
    <template v-if="tab === '超额归因'">
      <div v-if="!qa" class="card"><span class="spin"></span> 逐日分解中（约 10 秒）…</div>
      <template v-else>
        <div class="grid g5" style="margin-top:14px">
          <div class="kpi"><span>年化超额（对{{ bname }}）</span><b>{{ pct(qa.summary['年化超额']) }}</b><span>几何口径</span></div>
          <div class="kpi"><span>其中：幸存者偏差</span><b class="bad">{{ pct(qa.summary['分项年化']['幸存者偏差']) }}</b><span>股票池 = 当前成分</span></div>
          <div class="kpi"><span>其中：成分股股息</span><b>{{ pct(qa.summary['分项年化']['成分股股息']) }}</b><span>指数是价格指数</span></div>
          <div class="kpi"><span>选股 + 行业 + 风格</span><b class="ok">{{ pct(qa.summary['分项年化']['选股（特异）'] + qa.summary['分项年化']['行业'] + qa.summary['分项年化']['风格']) }}</b><span>模型贡献（费前）</span></div>
          <div class="kpi"><span>扣除幸存者偏差后</span><b>{{ pct(qa.summary['扣除幸存者偏差后年化超额']) }}</b><span>费用与成本 {{ pct(qa.summary['分项年化']['费用与交易成本']) }}</span></div>
        </div>
        <div class="card"><h3>超额分项累计（逐日相加）</h3>
          <LineChart :labels="qa.cum.dates" :base="0" :fmt="(v) => (v * 100).toFixed(0) + '%'"
            :series="[{ name: '超额', values: qa.cum['超额'], color: '#1b2430' }, { name: '选股（特异）', values: qa.cum['选股（特异）'], color: '#0E7C66' },
                      { name: '幸存者偏差', values: qa.cum['幸存者偏差'], color: '#c0392b' }, { name: '成分股股息', values: qa.cum['成分股股息'], color: '#d4a017' },
                      { name: '费用与交易成本', values: qa.cum['费用与交易成本'], color: '#888' }]" />
          <p class="sub">r_产品 − r_指数 = 费用与交易成本 + 行业 + 风格 + 选股（特异）+ 现金拖累 + 幸存者偏差 + 成分股股息，逐日严格相加。
            幸存者偏差 = 当前成分股组合（不含股息）− 指数：股票池用的是今天的中证 800，期间因上涨而调入中证500 的股票被提前「持有」了。配置 Tushare 历史成分权重后这一项会接近 0。</p></div>
        <div class="grid g2">
          <div class="card"><h3>分项年化</h3>
            <div v-for="(v, k) in qa.summary['分项年化']" :key="k" class="row" style="gap:6px;margin:3px 0"><span style="width:110px">{{ k }}</span>
              <div :style="{ width: Math.abs(v) * 2500 + 'px', height: '9px', background: v < 0 ? 'var(--red, #c0392b)' : 'var(--accent)', borderRadius: '3px' }"></div><span class="muted">{{ pct(v, 2) }}</span></div>
            <h3 style="margin-top:12px">风格分项年化</h3>
            <div v-for="(v, k) in qa.styles" :key="k" class="row" style="gap:6px;margin:2px 0"><span style="width:110px">{{ k }}</span><span :class="v < 0 ? 'neg' : ''">{{ pct(v, 2) }}</span></div></div>
          <div class="card"><h3>风险模型检验（事前 vs 事后跟踪误差）</h3>
            <p class="sub">偏差统计量 = 事后 / 事前，理想值 1。中位数 {{ median(qa.bias.map((b) => b['偏差统计量'])).toFixed(2) }}；调仓时已用过去的偏差对模型做了放大校正。</p>
            <div class="timeline" style="max-height:260px"><table><tr><th>调仓</th><th>事前</th><th>事后</th><th>偏差</th></tr>
              <tr v-for="b in qa.bias.slice().reverse()" :key="b.date"><td>{{ b.date }}</td><td>{{ pct(b['事前跟踪误差'], 1) }}</td><td>{{ pct(b['事后跟踪误差'], 1) }}</td>
                <td :class="b['偏差统计量'] > 1.5 ? 'bad' : ''">{{ b['偏差统计量'].toFixed(2) }}</td></tr></table></div></div>
        </div>
        <div class="card"><h3>逐月</h3>
          <table><tr><th>月份</th><th v-for="k in qaCols" :key="k">{{ k }}</th></tr>
            <tr v-for="r in qa.monthly.slice().reverse()" :key="r.month"><td>{{ r.month }}</td><td v-for="k in qaCols" :key="k" :class="r[k] < 0 ? 'neg' : ''">{{ pct(r[k], 2) }}</td></tr></table></div>
      </template>
    </template>

    <!-- 量化：因子研究 -->
    <template v-if="tab === '因子研究' && qr">
      <div class="card" style="margin-top:14px"><h3>单因子与合成因子检验（中证 800，月度，行业市值中性化）</h3>
        <table><tr><th>因子</th><th>类别</th><th>RankIC 均值</th><th>ICIR</th><th>t 值</th><th>IC&gt;0 占比</th><th>多空年化</th><th>月数</th><th>当前权重</th></tr>
          <tr v-for="f in qr.summary" :key="f.key" :style="f.key.startsWith('ALPHA') ? 'font-weight:600' : ''"><td>{{ f['因子'] }}</td><td>{{ f['类别'] }}</td>
            <td :class="f['RankIC 均值'] < 0 ? 'neg' : ''">{{ f['RankIC 均值'].toFixed(3) }}</td><td>{{ f['ICIR'].toFixed(2) }}</td>
            <td :class="Math.abs(f['t 值']) >= 2 ? 'ok' : ''">{{ f['t 值'].toFixed(2) }}</td><td>{{ pct(f['IC>0 占比'], 0) }}</td>
            <td :class="f['多空年化'] < 0 ? 'neg' : ''">{{ pct(f['多空年化'], 1) }}</td><td>{{ f['月数'] }}</td>
            <td>{{ qr.weights[f.key] != null ? pct(qr.weights[f.key], 0) : '' }}</td></tr></table>
        <p class="sub">合成权重 = 最近 12 期 RankIC 的 ICIR（负值置 0），只用远期窗口已结束的 IC，所以合成因子的 IC 是样本外的。等权合成作对照：本样本里它更好（估计出来的权重带噪声），
          但产品用哪种方法是事先定的，不按回测结果挑。单因子的 t 值都不到 2——用公开的低频数据做 A 股选股，信号本来就弱。</p></div>
      <div class="card"><h3>累计 RankIC</h3>
        <LineChart :labels="qr.cum_rank_ic.dates" :base="0" :fmt="(v) => v.toFixed(2)"
          :series="[{ name: 'ICIR 合成', values: qr.cum_rank_ic.ALPHA, color: '#1b2430', width: 2 }, { name: '等权合成', values: qr.cum_rank_ic.ALPHA_EW, color: '#0E7C66' },
                    { name: '盈利收益率', values: qr.cum_rank_ic.EP, color: '#d4a017' }, { name: '低波动', values: qr.cum_rank_ic.LOWVOL, color: '#6c5ce7' },
                    { name: '12-1 月动量', values: qr.cum_rank_ic.MOM, color: '#c0392b' }]" /></div>
      <div class="card"><h3>因子相关性（截面 Spearman，月度平均）</h3>
        <table><tr><th></th><th v-for="f in qr.corr.factors" :key="f" style="font-size:11px">{{ f }}</th></tr>
          <tr v-for="(row, i) in qr.corr.values" :key="i"><td style="font-size:11px">{{ qr.corr.factors[i] }}</td>
            <td v-for="(v, j) in row" :key="j" :style="{ background: i === j ? '#f3f4f6' : `rgba(${v > 0 ? '14,124,102' : '192,57,43'},${Math.min(Math.abs(v), 1) * 0.6})`, fontSize: '11px' }">{{ v.toFixed(2) }}</td></tr></table></div>
    </template>

    <!-- 量化：组合优化与执行 -->
    <template v-if="tab === '组合优化与执行' && qb">
      <div class="grid g2" style="margin-top:14px">
        <div class="card"><h3>最近目标组合（{{ qb.latest.date }}，{{ qb.latest.n }} 只）主动权重</h3>
          <table><tr><th>名称</th><th>行业</th><th>权重</th><th>基准</th><th>主动</th></tr>
            <tr v-for="t in qb.latest.top_active" :key="t.code"><td>{{ t.name }} <span class="muted">{{ t.code }}</span></td><td>{{ t.industry }}</td>
              <td>{{ pct(t.weight, 2) }}</td><td>{{ pct(t.bench, 2) }}</td><td :class="t.active < 0 ? 'neg' : 'ok'">{{ pct(t.active, 2) }}</td></tr></table></div>
        <div class="card"><h3>主动暴露（风险模型）</h3>
          <div v-for="(v, k) in qb.latest.style_exposure" :key="k" class="row" style="gap:6px;margin:3px 0"><span style="width:60px">{{ k }}</span>
            <div :style="{ width: Math.abs(v) * 400 + 'px', height: '9px', background: v < 0 ? '#c0392b' : 'var(--accent)', borderRadius: '3px' }"></div><span class="muted">{{ v.toFixed(3) }}</span></div>
          <p class="sub">风格暴露上限 ±0.30，规模单独 ±0.10（不靠小盘风格赚超额）；行业偏离按合同 3% 的 80% 控制。</p>
          <h3>行业主动权重（最大的 6 个）</h3>
          <div v-for="[k, v] in topIndustry" :key="k" class="row" style="gap:6px;margin:2px 0"><span style="width:80px">{{ k }}</span><span :class="v < 0 ? 'neg' : ''">{{ pct(v, 2) }}</span></div></div>
      </div>
      <div class="card"><h3>调仓记录（月末优化 → 次日起按成交量 10% 分日执行）</h3>
        <table><tr><th>信号日</th><th>状态 / 调整</th><th>事前 TE</th><th>预期超额</th><th>换手</th><th>持股</th><th>成分股</th><th>完成率</th><th>冲击成本</th></tr>
          <tr v-for="b in qb.rebalances.slice().reverse()" :key="b.date"><td>{{ b.date }}</td>
            <td style="text-align:left;font-size:11px">{{ b.status }}{{ b.relaxed?.length ? '：' + b.relaxed.join('；') : '' }}</td>
            <td>{{ pct(b.ex_ante_te, 1) }}</td><td>{{ pct(b.exp_alpha, 1) }}</td><td>{{ pct(b.turnover, 0) }}</td><td>{{ b.n_holdings }}</td>
            <td>{{ pct(b.member_weight, 0) }}</td><td :class="b.fill_rate < 0.95 ? 'bad' : ''">{{ pct(b.fill_rate, 1) }}</td><td>{{ b.impact_bp?.toFixed(1) }} bp</td></tr></table></div>
    </template>

    <!-- 业绩归因 -->
    <template v-if="tab === '业绩归因' && att">
      <div class="card"><h3>Brinson 归因（累计，各月加总）</h3>
        <table><tr><th v-for="k in attCols" :key="k">{{ k }}</th></tr><tr><td v-for="k in attCols" :key="k">{{ pct(att.brinson_total[k]) }}</td></tr></table>
        <p class="sub">基准总收益 = {{ bname }} 实际涨跌；基准行业权重与收益 = 月初{{ bname }} 成分股按申万一级归类（权重来源：{{ att.brinson?.[0]?.基准权重来源 }}）。成分股拼出的基准与真实指数之差（分红再投资、幸存者偏差）单列「基准近似误差」；配置 Tushare token 后自动改用历史成分权重。</p></div>
      <div class="card"><h3>逐月</h3>
        <table><tr><th>月份</th><th v-for="k in attCols" :key="k">{{ k }}</th><th>超配最多</th><th>低配最多</th></tr>
          <tr v-for="b in att.brinson.slice().reverse()" :key="b.month"><td>{{ b.month }}</td><td v-for="k in attCols" :key="k" :class="b[k] < 0 ? 'neg' : ''">{{ pct(b[k]) }}</td><td>{{ b['超配最多'] }}</td><td>{{ b['低配最多'] }}</td></tr></table></div>
      <div class="card"><h3>风格暴露（中证 800 截面 z 分数）</h3>
        <table><tr><th>因子</th><th>组合</th><th>基准</th><th>主动暴露</th></tr>
          <tr v-for="s in att.style" :key="s.factor"><td>{{ s.factor }}</td><td>{{ s['组合'].toFixed(2) }}</td><td>{{ s['基准'].toFixed(2) }}</td><td :class="Math.abs(s['主动暴露']) > 0.5 ? 'bad' : ''">{{ s['主动暴露'].toFixed(2) }}</td></tr></table></div>
    </template>

    <!-- 股票池与投委会 -->
    <template v-if="tab === '股票池与投委会' && pool">
      <div class="grid g2">
        <div class="card"><h3>核心池（{{ pool.pools.filter((p) => p.pool === '核心池').length }}）</h3>
          <table><tr><th>名称</th><th>代码</th><th>入池日</th><th>依据</th></tr>
            <tr v-for="p in pool.pools.filter((p) => p.pool === '核心池')" :key="p.code"><td>{{ p.name }}</td><td>{{ p.code }}</td><td>{{ p.since }}</td><td class="sub" style="text-align:left">{{ p.reason }}</td></tr></table>
          <h3 style="margin-top:12px">禁投池</h3><div v-for="b in pool.banned" :key="b.code" class="sub">{{ b.code }}：{{ b.reason }}</div></div>
        <div class="card"><h3>提交投委会</h3>
          <div class="row"><input v-model="icForm.code" placeholder="代码" style="width:90px" /><input v-model="icForm.name" placeholder="名称" style="width:100px" />
            <select v-model="icForm.request"><option>入核心池</option><option>调出核心池</option><option>入禁投池</option></select></div>
          <div class="row" style="margin-top:6px"><input v-model="icForm.material" placeholder="申请材料 / 研究备忘录链接" style="flex:1" /></div>
          <div class="row" style="margin-top:6px"><select v-model="icForm.decision"><option>批准</option><option>否决</option><option>补充材料</option></select>
            <input v-model="icForm.note" placeholder="决议说明" style="flex:1" /><button @click="submitIC">记录决议</button></div>
          <h3 style="margin-top:12px">最近决议</h3>
          <div class="timeline"><div v-for="d in pool.ic.slice(0, 30)" :key="d.id"><span class="t" style="width:80px">{{ d.decided_at }}</span>
            <span class="lab">{{ d.decision }}</span>{{ d.name }}（{{ d.code }}）{{ d.request }} · {{ d.requested_by }}</div></div></div>
      </div>
    </template>

    <!-- 交易前检查 -->
    <template v-if="tab === '交易前检查'">
      <div class="card"><h3>拟下单指令（每行：代码 买/卖 数量）</h3>
        <textarea v-model="orderText" rows="5" style="width:100%;font-family:monospace;border:1px solid var(--line);border-radius:6px;padding:8px"></textarea>
        <div class="row" style="margin-top:6px"><button @click="checkOrders">检查</button><span class="sub">按最新估值日收盘价模拟成交，检查禁投池、核心池、预警线、单票、行业、现金</span></div></div>
      <div class="card" v-if="pre"><h3>结果：<span :class="pre.pass ? 'ok' : 'bad'">{{ pre.pass ? '全部通过' : '不通过' }}</span></h3>
        <table><tr><th>代码</th><th>方向</th><th>数量</th><th>价格</th><th>结果</th><th>原因</th></tr>
          <tr v-for="o in pre.orders" :key="o.code + o.side"><td>{{ o.code }}</td><td>{{ o.side === 'buy' ? '买' : '卖' }}</td><td>{{ o.qty }}</td><td>{{ o.price?.toFixed(2) }}</td>
            <td :class="o.pass ? 'ok' : 'bad'">{{ o.pass ? '通过' : '拒绝' }}</td><td style="text-align:left">{{ o.reasons.join('；') }}</td></tr></table>
        <div v-for="c in pre.portfolio_checks" :key="c" class="bad" style="margin-top:4px">组合层面：{{ c }}</div></div>
    </template>

    <!-- 估值表与对账 -->
    <template v-if="tab === '估值表与对账'">
      <div class="grid g2">
        <div class="card"><h3>估值核对（托管净值 vs 自估净值）</h3>
          <table><tr><th>日期</th><th>差异</th><th>解释项</th><th>状态</th></tr>
            <tr v-for="x in ov.reconciliation" :key="x.date"><td>{{ x.date }}</td><td>{{ x.diff.toFixed(2) }}</td>
              <td style="text-align:left" class="sub">{{ Object.entries(typeof x.explained === 'string' ? JSON.parse(x.explained) : x.explained).map(([k, v]) => `${k} ${v}`).join('；') }}</td>
              <td><span :class="['tagx', x.status === '一致' ? '正常' : '超限']">{{ x.status }}</span></td></tr></table></div>
        <div class="card"><h3>托管估值表（最近 30 个交易日）</h3>
          <div v-for="s in sheets" :key="s.name"><a :href="s.url">{{ s.name }}</a></div></div>
      </div>
      <div class="card"><h3>投资者月报</h3>
        <div class="row"><input v-model="month" style="width:100px" /><button :disabled="reporting" @click="makeReport"><span v-if="reporting" class="spin"></span>生成月报</button>
          <a v-if="reportUrl" :href="reportUrl" target="_blank">打开 PDF</a></div></div>
    </template>
  </template>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { api, followJob, pct, wan } from '../api'
import LineChart from '../components/LineChart.vue'
import PEView from './PE.vue'

const isQuant = computed(() => ov.value?.product.type === 'quant_enhanced')
const isPE = computed(() => products.value.find((p) => p.code === code.value)?.type === 'private_equity')
const initialTab = ref('')
const bname = computed(() => ov.value?.product.bench_name || '基准')
const tabs = computed(() => isQuant.value ? ['产品概览', '风控合规', '超额归因', '因子研究', '组合优化与执行', '估值表与对账']
  : ['产品概览', '风控合规', '业绩归因', '股票池与投委会', '交易前检查', '估值表与对账'])
const qa = ref(null), qr = ref(null), qb = ref(null)
const qaCols = ['产品', '指数', '超额', '费用与交易成本', '行业', '风格', '选股（特异）', '现金拖累', '幸存者偏差', '成分股股息']
const topIndustry = computed(() => Object.entries(qb.value?.latest.industry_exposure || {}).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 6))
const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : 0 }
const products = ref([]), code = ref('CV-VG1'), tab = ref('产品概览')
const ov = ref(null), events = ref([]), att = ref(null), pool = ref(null), sheets = ref([]), pre = ref(null)
const m = computed(() => ov.value?.metrics), r = computed(() => ov.value?.risk || {})
const attCols = ['组合收益', '基准收益', '超额', '配置效应', '选股效应', '交互效应', '其他（交易与现金）', '基准近似误差']
const orderText = ref('000002 买 10000\n600519 买 100\n600030 买 1000')
const icForm = ref({ code: '', name: '', request: '入核心池', material: '', decision: '批准', note: '' })
const month = ref('2026-09'), reporting = ref(false), reportUrl = ref('')

onMounted(async () => {   // 支持深链接：/fund?product=CV-QE1&tab=超额归因
  const q = new URLSearchParams(location.search)
  initialTab.value = q.get('tab') || ''                 // 先设好：PE 页面在产品列表加载后立刻挂载，挂载时就要拿到 tab
  if (q.get('product')) code.value = q.get('product')
  products.value = await api('/api/fund/products')
  await loadAll()
  if (q.get('tab') && tabs.value.includes(q.get('tab'))) tab.value = q.get('tab')
})
async function loadAll() {
  if (isPE.value) { ov.value = null; return }
  ov.value = await api(`/api/fund/${code.value}/overview`)
  events.value = await api(`/api/fund/${code.value}/events`)
  sheets.value = await api(`/api/fund/${code.value}/sheets`)
  att.value = qa.value = qr.value = qb.value = null
  if (isQuant.value) {
    api(`/api/quant/${code.value}/rebalances`).then((x) => (qb.value = x))
    api('/api/quant/research').then((x) => (qr.value = x))
    api(`/api/quant/${code.value}/attribution`).then((x) => (qa.value = x))
  } else {
    pool.value = await api(`/api/fund/${code.value}/pool`)
    api(`/api/fund/${code.value}/attribution`).then((x) => (att.value = x))
  }
}
async function handle(e) {
  const note = prompt(`处理说明（${e.detail}，${e.start} ~ ${e.end}）`, '已知悉，下次调仓降低行业集中度'); if (!note) return
  await api(`/api/fund/${code.value}/events/handle`, { ids: e.ids, handled_by: '风控岗', note })
  events.value = await api(`/api/fund/${code.value}/events`); ov.value.open_events = events.value.length
}
async function checkOrders() {
  const orders = orderText.value.trim().split('\n').map((l) => l.trim().split(/\s+/)).filter((x) => x.length >= 3)
    .map(([c, s, q]) => ({ code: c, side: s.startsWith('买') || s === 'buy' ? 'buy' : 'sell', qty: Number(q) }))
  pre.value = await api(`/api/fund/${code.value}/pretrade`, { orders })
}
async function submitIC() { await api(`/api/fund/${code.value}/ic`, { ...icForm.value, votes: { 赞成: 3, 反对: 0, 弃权: 0 } }); pool.value = await api(`/api/fund/${code.value}/pool`) }
async function makeReport() {
  reporting.value = true; reportUrl.value = ''
  try { const { job_id } = await api(`/api/fund/${code.value}/report`, { month: month.value }); const res = await followJob(job_id); reportUrl.value = res.pdf || res.html }
  finally { reporting.value = false }
}
</script>
