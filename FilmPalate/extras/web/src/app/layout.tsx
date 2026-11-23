import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { Providers } from "@/components/providers";
import { SiteHeader } from "@/components/site-header";
import { Toaster } from "@/components/ui/sonner";

import "./globals.css";

const geistSans = Geist({ variable: "--font-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "FilmPalate",
  description: "Movie picks matched to your palate: rate what you've seen, save what you want to watch.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`dark ${geistSans.variable} ${geistMono.variable} h-full antialiased`} style={{ colorScheme: "dark" }}>
      <body className="fp-backdrop flex min-h-full flex-col text-foreground">
        <Providers>
          <SiteHeader />
          <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">{children}</main>
          <footer className="border-t py-6 text-center text-xs text-muted-foreground">
            FilmPalate · DTSC 3602 class project · Model: List taste model (Ishika Bhattarai) · Ratings data: MovieLens (GroupLens) · Movie info: IMDb ·
            Posters: TMDB (this product uses the TMDB API but is not endorsed or certified by TMDB)
          </footer>
        </Providers>
        <Toaster theme="dark" />
      </body>
    </html>
  );
}
