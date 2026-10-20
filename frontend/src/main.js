import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import Risk from './views/Risk.vue'
import Research from './views/Research.vue'
import Filings from './views/Filings.vue'
import Calls from './views/Calls.vue'
import Fund from './views/Fund.vue'
import './style.css'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/fund' },
    { path: '/fund', component: Fund, meta: { title: '资管中台' } },
    { path: '/risk', component: Risk, meta: { title: '组合风控' } },
    { path: '/research', component: Research, meta: { title: '研究备忘录' } },
    { path: '/filings', component: Filings, meta: { title: '年报问答' } },
    { path: '/calls', component: Calls, meta: { title: '业绩会纪要' } },
  ],
})
createApp(App).use(router).mount('#app')
