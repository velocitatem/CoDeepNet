import { redirect } from 'next/navigation'
import { createClient } from '@/utils/supabase/server'
import { logout } from './actions'

export default async function DashboardPage() {
  const supabase = await createClient()

  const { data, error } = await supabase.auth.getUser()
  if (error || !data?.user) {
    redirect('/login')
  }

  return (
    <div>
      <p>Welcome, {data.user.email}</p>
      <form>
        <button formAction={logout}>Logout</button>
      </form>
    </div>
  )
}