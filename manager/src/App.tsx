import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { Data } from './pages/Data'
import { Live } from './pages/Live'
import { Models } from './pages/Models'
import { Outputs } from './pages/Outputs'
import { Pipelines } from './pages/Pipelines'
import { Tunes } from './pages/Tunes'

// One entry per page. When adding a backend feature, add (or extend) a page here so the
// manager stays the single place to see and control everything (see AGENTS.md).
const PAGES = [
  { path: '/live', label: 'Live', el: <Live /> },
  { path: '/data', label: 'Data', el: <Data /> },
  { path: '/tunes', label: 'Tunes', el: <Tunes /> },
  { path: '/pipelines', label: 'Pipelines', el: <Pipelines /> },
  { path: '/models', label: 'Models', el: <Models /> },
  { path: '/outputs', label: 'Outputs', el: <Outputs /> },
]

export default function App() {
  return (
    <>
      <nav>
        <span className="brand">🎵 Whistle House</span>
        {PAGES.map((p) => (
          <NavLink key={p.path} to={p.path}>
            {p.label}
          </NavLink>
        ))}
      </nav>
      <main>
        <Routes>
          {PAGES.map((p) => (
            <Route key={p.path} path={p.path} element={p.el} />
          ))}
          <Route path="*" element={<Navigate to="/live" replace />} />
        </Routes>
      </main>
    </>
  )
}
