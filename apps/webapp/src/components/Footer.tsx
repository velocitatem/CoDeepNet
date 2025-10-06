import Link from "next/link";
import { getLocale } from "@/libs/locales";

export default function Footer() {
  const { common } = getLocale('en');
  
  return (
    <footer>
      {/* TODO: Style this footer when implementing in your project */}
      <div>
        <div>
          <div>
            <h3>{common.footer.brand}</h3>
            <p>{common.footer.description}</p>
          </div>
          <div>
            <h4>{common.footer.legal.title}</h4>
            <ul>
              <li><Link href="/privacy-policy">{common.footer.legal.privacyPolicy}</Link></li>
              <li><Link href="/tos">{common.footer.legal.termsOfService}</Link></li>
            </ul>
          </div>
          <div>
            <h4>{common.footer.company.title}</h4>
            <ul>
              <li><Link href="/blog">{common.footer.company.blog}</Link></li>
              <li><Link href="/dashboard">{common.footer.company.dashboard}</Link></li>
            </ul>
          </div>
        </div>
        <div>
          <p>{common.footer.copyright}</p>
        </div>
      </div>
    </footer>
  );
}