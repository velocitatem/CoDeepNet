export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div>
      <nav>
        <h1>Dashboard</h1>
      </nav>
      <main>{children}</main>
    </div>
  )
}