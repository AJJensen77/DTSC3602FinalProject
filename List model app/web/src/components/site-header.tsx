"use client";

import { Wine } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";

import { Button } from "@/components/ui/button";
import { supabase } from "@/lib/supabase";
import { useLists, useSession } from "@/lib/use-lists";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: "/for-you", label: "For you" },
  { href: "/browse", label: "Browse" },
  { href: "/my-lists", label: "My lists" },
];

export function SiteHeader() {
  const { session } = useSession();
  const { rated, want } = useLists();
  const pathname = usePathname();
  const router = useRouter();

  async function signOut() {
    await supabase.auth.signOut();
    router.replace("/");
  }

  return (
    <header className="sticky top-0 z-20 border-b bg-background/90 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-4 px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <Wine className="size-5 text-primary" aria-hidden />
          FilmPalate
        </Link>
        {(
          <nav className="flex gap-1 overflow-x-auto">
            {LINKS.filter((l) => session || l.href === "/browse").map((l) => (
              <Link
                key={l.href}
                href={l.href}
                className={cn(
                  "whitespace-nowrap rounded-md px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground",
                  pathname === l.href && "bg-muted font-medium text-foreground",
                )}
              >
                {l.label}
                {l.href === "/my-lists" && rated.length + want.length > 0 && (
                  <span className="ml-1 text-xs tabular-nums">({rated.length + want.length})</span>
                )}
              </Link>
            ))}
          </nav>
        )}
        <div className="ml-auto flex items-center gap-2">
          {session ? (
            <>
              <span className="hidden text-sm text-muted-foreground md:inline">{session.user.email}</span>
              <Button variant="outline" size="sm" onClick={signOut}>
                Sign out
              </Button>
            </>
          ) : (
            <>
              <Button variant="ghost" size="sm" nativeButton={false} render={<Link href="/login" />}>
                Log in
              </Button>
              <Button size="sm" nativeButton={false} render={<Link href="/signup" />}>
                Sign up
              </Button>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
