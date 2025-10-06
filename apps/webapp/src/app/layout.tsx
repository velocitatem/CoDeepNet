import type { Metadata } from "next";
import "./globals.css";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

const fontVariables = "font-sans";

export const metadata: Metadata = {
    title: "Ultiplate - Ultimate Boilerplate",
    description: "AI-native template for any project with deployment ready setup",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <body className={`${fontVariables} antialiased`}>
                <Header />
                <main className="min-h-screen">
                    {children}
                </main>
                <Footer />
            </body>
        </html>
    );
}
