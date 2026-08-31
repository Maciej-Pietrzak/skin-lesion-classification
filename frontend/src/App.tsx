import { Link, Route, Routes } from 'react-router-dom'
import { Toaster } from 'sonner'
import { CaseDetailPage } from './pages/CaseDetailPage'
import { CaseHistoryPage } from './pages/CaseHistoryPage'
import { UploadPredictPage } from './pages/UploadPredictPage'

function App() {
  return (
    <div>
      <Toaster richColors position="top-right" />
      <nav className="flex gap-4 border-b border-gray-200 p-4">
        <Link to="/">Nowa analiza</Link>
        <Link to="/cases">Historia przypadków</Link>
      </nav>
      <main className="p-4">
        <Routes>
          <Route path="/" element={<UploadPredictPage />} />
          <Route path="/cases" element={<CaseHistoryPage />} />
          <Route path="/cases/:id" element={<CaseDetailPage />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
