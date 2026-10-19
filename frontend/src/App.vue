<template>
  <div class="shell">
    <aside class="nav">
      <div class="brand"><span class="mark">◆</span> CYRON <b>VERITAS</b></div>
      <div class="tag">AI 投研与风控助手</div>
      <router-link to="/risk"><span>◧</span> 组合风控</router-link>
      <router-link to="/research"><span>◩</span> 研究备忘录</router-link>
      <router-link to="/filings"><span>◨</span> 年报问答</router-link>
      <router-link to="/calls"><span>◪</span> 业绩会纪要</router-link>
      <div class="foot">数字由代码计算 · 结论可追溯<br />不构成投资建议</div>
    </aside>
    <main class="main"><router-view v-slot="{ Component }"><keep-alive><component :is="Component" /></keep-alive></router-view></main>
  </div>
</template>
