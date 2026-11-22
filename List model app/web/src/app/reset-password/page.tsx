"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { supabase } from "@/lib/supabase";

/**
 * Where the reset email's link lands. Supabase signs the person in from the link
 * (a short-lived "recovery" session), then they choose a new password here.
 */
export default function ResetPasswordPage() {
  const router = useRouter();
  // checking -> ready (signed in from the link) | invalid (expired / other device)
  const [status, setStatus] = useState<"checking" | "ready" | "invalid">("checking");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const { data } = supabase.auth.onAuthStateChange((event, session) => {
      if (session && (event === "PASSWORD_RECOVERY" || event === "SIGNED_IN" || event === "INITIAL_SESSION")) {
        setStatus("ready");
      }
    });
    // The link's code is exchanged for a session on page load; give it a moment.
    const t = setTimeout(async () => {
      const { data: s } = await supabase.auth.getSession();
      setStatus((cur) => (cur === "ready" || s.session ? "ready" : "invalid"));
    }, 2500);
    return () => {
      data.subscription.unsubscribe();
      clearTimeout(t);
    };
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (password.length < 8) return setError("Use at least 8 characters for your password.");
    if (password !== confirm) return setError("The two passwords don't match.");
    setBusy(true);
    const { error } = await supabase.auth.updateUser({ password });
    setBusy(false);
    if (error) {
      setError(/different from the old/i.test(error.message)
        ? "Please choose a password different from your old one."
        : error.message);
      return;
    }
    toast.success("Password updated. You're logged in.");
    router.replace("/for-you");
  }

  return (
    <Card className="mx-auto w-full max-w-sm">
      <CardHeader>
        <CardTitle>Choose a new password</CardTitle>
        <CardDescription>You&apos;ll stay logged in after saving it.</CardDescription>
      </CardHeader>
      <CardContent>
        {status === "checking" && <p className="text-sm text-muted-foreground">Checking your reset link…</p>}
        {status === "invalid" && (
          <div className="flex flex-col gap-4">
            <Alert variant="destructive">
              <AlertDescription>
                This reset link has expired, was already used, or was opened on a different device or browser than the one
                you requested it from. Request a new one and open it on the same device.
              </AlertDescription>
            </Alert>
            <Button nativeButton={false} render={<Link href="/forgot-password" />}>Send a new link</Button>
          </div>
        )}
        {status === "ready" && (
          <form onSubmit={submit} className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <Label htmlFor="password">New password</Label>
              <Input id="password" type="password" autoComplete="new-password" required value={password}
                     onChange={(e) => setPassword(e.target.value)} />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="confirm">Confirm new password</Label>
              <Input id="confirm" type="password" autoComplete="new-password" required value={confirm}
                     onChange={(e) => setConfirm(e.target.value)} />
            </div>
            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            <Button type="submit" disabled={busy}>{busy ? "Saving…" : "Save new password"}</Button>
          </form>
        )}
      </CardContent>
    </Card>
  );
}
